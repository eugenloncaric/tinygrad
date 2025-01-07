from dataclasses import dataclass
from tinygrad.dtype import dtypes
from tinygrad.ops import UOp, Ops, GroupOp, Variable, PatternMatcher, UPat, type_verify, graph_rewrite_map, graph_rewrite, track_rewrites, identity_element
from tinygrad.ops import merge_views, symbolic_simple, view_left
from tinygrad.device import Buffer
from tinygrad.helpers import Metadata, all_int, unwrap, prod
from tinygrad.shape.shapetracker import ShapeTracker

# ** tensor uop spec

tensor_uop_spec = PatternMatcher([
  (UPat(Ops.DEVICE, dtypes.void, (), name="root"), lambda root: isinstance(root.arg, str)),
  (UPat(Ops.BUFFER, name="root", src=(UPat(Ops.DEVICE))), lambda root: isinstance(root.arg, tuple) and all_int(root.arg) and len(root.arg) == 2),
  (UPat(Ops.COPY, name="root", src=(UPat(Ops.DEVICE), UPat.var("copyin"))), lambda root,copyin: isinstance(root.arg, bool) and copyin.dtype == root.dtype),
  (UPat(GroupOp.Movement, name="root", src=(UPat(),)), lambda root: isinstance(root.arg, tuple)),
  (UPat({Ops.CONTIGUOUS, Ops.DETACH}, src=(UPat(),)), lambda: True),
])

# ** ScheduleItem return type

@dataclass
class ScheduleItem:
  ast: UOp
  bufs: tuple[Buffer, ...]
  metadata: tuple[Metadata, ...]

# ** scheduler rewrite

remove_movement_ops = merge_views+PatternMatcher([
  (UPat(GroupOp.Movement, name="mov", src=(UPat.var("x"),)), lambda mov,x: x.view(mov.st)),
  (UPat(Ops.VIEW, name="view", src=(UPat.var("x"),)), lambda view,x: x if view.st.contiguous and x.st is not None and x.shape == view.shape else None),
  # const is free to copy around, so this view just merges
  (UPat(Ops.VIEW, name="vm1", src=(UPat(Ops.CONST, name="x", src=(UPat(Ops.VIEW, name="vm2"),)),)), lambda x,vm1,vm2: x.replace(src=(vm2.view(vm1.st),))),
  # masked const becomes a valid, this structurally preventrs const folding.
  (UPat(Ops.CONST, name="root", src=(UPat(Ops.VIEW, name="view"),)),
   lambda root,view: None if view.st.views[0].mask is None else root.valid())
])

def collapse_size0_op(root:UOp):
  if root.base.st is None or root.size != 0: return None
  if root.base.op is Ops.CONST and root.const_arg == 0: return None
  return root.const_like(0)

def collapse_const_reduce(root:UOp, x:UOp):
  prshape = prod(unwrap(x.st).shape[i] for i in root.arg[1])
  ret = x.const_arg
  match root.arg[0]:
    case Ops.ADD: ret *= prshape
    case Ops.MUL: ret **= prshape
    case Ops.MAX: pass # NOTE: Ops.MAX is passthrough
    case _: return None
  return root.const_like(ret)

sym = symbolic_simple+PatternMatcher([
  (UPat(set(Ops), name="root"), collapse_size0_op),

  # reduce folding
  (UPat(Ops.REDUCE_AXIS, name="root", src=(UPat(Ops.CONST, arg=0),)),
   lambda root: root.const_like(identity_element(root.arg[0], root.dtype)) if root.size != 0 else None),
  (UPat(Ops.REDUCE_AXIS, name="root", src=(UPat(Ops.CONST, name="x"),)), collapse_const_reduce),

  # copy folding
  (UPat(Ops.COPY, src=(UPat(), UPat.cvar("x"),)), lambda x:x),

  # detach is a noop here
  (UPat(Ops.DETACH, src=(UPat.var("x"))), lambda x:x),
])

def add_buffer(ctx:dict[UOp, UOp], root:UOp):
  if root.base.st is None or root.base.op in {Ops.BUFFER, Ops.CONST, Ops.VALID, Ops.VIEW}: return None
  buffer = UOp.new_buffer(root.device, root.size, root.dtype)
  ctx[buffer] = root
  return buffer.view(unwrap(root.st))

bufferize = PatternMatcher([
  (UPat(set(Ops)-{Ops.SINK}, name="root"), add_buffer),
])

# ** ast rewrite

def add_loads(ctx:list[UOp], buf:UOp):
  if buf not in ctx: ctx.append(buf)
  glbl = UOp(Ops.DEFINE_GLOBAL, buf.dtype.ptr(size=buf.size), (), ctx.index(buf))
  return UOp(Ops.LOAD, buf.dtype.base, (glbl, unwrap(buf.st).to_uop()))

def add_stores(sink:UOp):
  if all(x.op is Ops.STORE for x in sink.src): return None
  new_src: list[UOp] = []
  for i,x in enumerate(sink.src):
    glbl = UOp(Ops.DEFINE_GLOBAL, x.dtype.ptr(size=x.size), (), i)
    new_src.append(UOp.store(glbl, ShapeTracker.from_shape(x.shape).to_uop(), x))
  return UOp.sink(*new_src)

debufferize = PatternMatcher([
  (UPat(Ops.BUFFER, name="buf"), add_loads),
  (UPat(Ops.SINK, name="sink"), add_stores),
])

to_si = PatternMatcher([
  (UPat(Ops.SINK, src=(UPat.store(UPat(), UPat(), UPat(GroupOp.Meta, name="meta")))), lambda meta:meta),
  (UPat(Ops.CONTIGUOUS, src=(UPat.var("x"),)), lambda x:x),
  (UPat(Ops.CONST, name="root", src=(UPat(),)), lambda root:root.replace(src=())),
])

# ** schedule creation

@track_rewrites(named=True)
def create_schedule_with_vars(outs:list[UOp]) -> tuple[list[ScheduleItem], dict[Variable, int], dict[UOp, UOp]]:
  # verify
  sink = UOp.sink(*outs)
  type_verify(list(sink.toposort), tensor_uop_spec)

  # create kernels from the schedule graph
  var_vals: dict[Variable, int] = {}
  realizes: dict[UOp, UOp] = {}
  tensor_map = graph_rewrite_map(sink, remove_movement_ops+sym)
  buffer_map = graph_rewrite_map(tensor_map[sink], remove_movement_ops+sym+bufferize, realizes)

  # remap
  rev_tensor_map: dict[UOp, list[UOp]] = {}
  for k,v in tensor_map.items():
    if v not in rev_tensor_map: rev_tensor_map[v] = []
    rev_tensor_map[v].append(k)
  tensor_buffers: dict[UOp, list[UOp]] = {}
  for k,v in buffer_map.items():
    if v.base.op is Ops.BUFFER:
      if v.base not in tensor_buffers: tensor_buffers[v.base] = []
      tensor_buffers[v.base].extend(rev_tensor_map[k])

  # schedule
  schedule: list[ScheduleItem] = []
  becomes_map: dict[UOp, UOp] = {}
  for k,v in realizes.items():
    ast = graph_rewrite(v.sink(), debufferize+view_left, bufs:=[k])
    schedule.append(ScheduleItem(graph_rewrite(ast, to_si), tuple(b.buffer for b in bufs), ()))
    # update buffer refs
    for b in bufs: b.buffer.ref(1)
    tensor_refs = tensor_buffers[k]
    for tr in tensor_refs:
      becomes_map[tr] = k.view(unwrap(tr.st))
  return schedule, var_vals, becomes_map
