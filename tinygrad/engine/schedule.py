from dataclasses import dataclass
from tinygrad.dtype import ImageDType, dtypes
from tinygrad.ops import UOp, Ops, GroupOp, Variable, PatternMatcher, UPat, type_verify, graph_rewrite, track_rewrites, identity_element, buffers
from tinygrad.ops import merge_views, symbolic_simple, view_left, graph_rewrite_map
from tinygrad.device import Buffer
from tinygrad.helpers import Metadata, DEBUG, all_int, unwrap, prod, dedup
from tinygrad.shape.shapetracker import ShapeTracker

BUF_LIMIT = {"METAL":32}

# ** tensor uop spec

tensor_uop_spec = PatternMatcher([
  (UPat(Ops.DEVICE, dtypes.void, (), name="root"), lambda root: isinstance(root.arg, str)),
  (UPat(Ops.BUFFER, name="root", src=(UPat(Ops.DEVICE))), lambda root: isinstance(root.arg, tuple) and all_int(root.arg) and len(root.arg) == 2),
  (UPat(GroupOp.Movement, name="root", src=(UPat(),)), lambda root: isinstance(root.arg, tuple)),

  (UPat(Ops.BIND, src=(UPat(Ops.DEFINE_VAR), UPat(Ops.CONST)), arg=None), lambda: True),
  (UPat({Ops.CONST, Ops.DEFINE_VAR}, src=(UPat(Ops.VIEW, src=(UPat(Ops.DEVICE),)),)), lambda: True),

  (UPat(Ops.COPY, name="copy", src=(UPat(Ops.DEVICE), UPat.var("x"))), lambda copy,x: isinstance(copy.arg, bool) and copy.dtype == x.dtype),
  (UPat(Ops.EMPTY, src=(UPat(Ops.VIEW, src=(UPat(Ops.BUFFER),)),), arg=None), lambda: True),
  (UPat(Ops.BUFFER_VIEW, src=(UPat(Ops.VIEW, src=(UPat(Ops.DEVICE),)), UPat()), arg=None), lambda: True),

  (UPat(Ops.ASSIGN, name="assign", src=(UPat.var("target"), UPat.var("new_val")), arg=None),
   lambda assign,target,new_val: (target.op is Ops.BUFFER or target.is_realized) and (assign.dtype == target.dtype == new_val.dtype)),
  (UPat((Ops.DETACH, Ops.CONTIGUOUS), name="root", src=(UPat.var("x"),), arg=None), lambda root,x: root.dtype == x.dtype),
])

# ** ScheduleItem return type

@dataclass
class ScheduleItem:
  ast: UOp
  bufs: tuple[Buffer, ...]
  metadata: tuple[Metadata, ...]

# ** movement ops rewrite rules

remove_movement_ops = merge_views+PatternMatcher([
  (UPat(GroupOp.Movement, name="mov", src=(UPat.var("x"),)), lambda mov,x: x.view(mov.st)),
  (UPat(Ops.VIEW, name="view", src=(UPat.var("x"),)), lambda view,x:x if view.st.contiguous and x.st is not None and x.shape == view.shape else None),
  # const is free to copy around, so this view just merges
  (UPat(Ops.VIEW, name="v2", src=(UPat(Ops.CONST, name="x", src=(UPat(Ops.VIEW, name="v1"),)),)), lambda x,v1,v2: x.replace(src=(v1.view(v2.st),))),
  # masked CONST becomes VALID, this structurally prevents future const folding
  (UPat(Ops.CONST, name="root", src=(UPat(Ops.VIEW, name="view"),)),
   lambda root,view: None if view.st.views[0].mask is None else root.valid())
])

# ** symbolic **

def collapse_size0_op(root:UOp):
  if root.base.st is None or root.size != 0: return None
  if root.base.op is Ops.CONST and root.const_arg == 0: return None
  return root.const_like(0)

def collapse_const_reduce(root:UOp, x:UOp):
  if not all_int(x.shape): return None
  prshape = prod(unwrap(x.st).shape[i] for i in root.arg[1])
  ret = x.const_arg
  match root.arg[0]:
    case Ops.ADD: ret *= prshape
    case Ops.MUL: ret **= prshape
    case Ops.MAX: pass # NOTE: Ops.MAX is passthrough
    case _: return None
  return root.const_like(ret)

def reorder_assigns(root:UOp):
  if len([x for x in root.src if x.op is Ops.ASSIGN]) == 0: return None
  # TODO: handle multiple diamond assigns
  new_src = sorted(root.src, key=lambda x:0 if x.op is Ops.ASSIGN else -1)
  return root.replace(src=tuple(new_src)) if tuple(new_src) != root.src else None

sym = symbolic_simple+PatternMatcher([
  (UPat(set(Ops), name="root"), collapse_size0_op),

  # reorder sources such that LOAD comes before ASSIGN
  (UPat(set(Ops), name="root"), reorder_assigns),

  # reduce folding
  (UPat(Ops.REDUCE_AXIS, name="root", src=(UPat(Ops.CONST, arg=0),)),
   lambda root: root.const_like(identity_element(root.arg[0], root.dtype)) if root.size != 0 else None),
  (UPat(Ops.REDUCE_AXIS, name="root", src=(UPat(Ops.CONST, name="x"),)), collapse_const_reduce),

  # copy folding
  (UPat(Ops.COPY, src=(UPat(), UPat.cvar("x"),)), lambda x:x),

  # detach is a noop here
  (UPat(Ops.DETACH, src=(UPat.var("x"))), lambda x:x),
])

# ** kernel creation

def create_buffer(ctx:dict[UOp, UOp], root:UOp):
  if root.base.st is None or root.base.op in {Ops.SINK, Ops.BIND, Ops.DEFINE_VAR, Ops.BUFFER, Ops.CONST, Ops.VALID, Ops.VIEW}: return None
  buffer = UOp.new_buffer(root.device, root.size, root.dtype)
  ctx[buffer] = root
  return buffer.view(unwrap(root.st))

def create_subbuffer(ctx:dict[UOp, UOp], root:UOp, src:UOp):
  sbuffer = UOp.new_buffer(root.device, root.size, root.dtype)
  buffers[sbuffer] = src.base.buffer.view(root.size, root.dtype, unwrap(src.st).views[0].offset*src.dtype.itemsize)
  ctx[sbuffer] = root
  return sbuffer.view(unwrap(root.st))

def add_target_buf(ctx:dict[UOp, UOp], root:UOp, target:UOp):
  ctx[target.base] = root
  return target

def view_src(ctx:dict[UOp, UOp], src:UOp, view:UOp):
  if src.op is Ops.BUFFER or src.st is None: return None
  if view.size <= src.size and all(v.mask is None for v in view.st.views): return None
  if (buffer_src:=create_buffer(ctx, src)) is None: return None
  return buffer_src.view(view.st)

def bufferize_input(ctx:dict[UOp, UOp], root:UOp, dest:UOp, x:UOp):
  if x.base.op is Ops.BUFFER: return None
  if (buffer_src:=create_buffer(ctx, x.base)) is None: raise RuntimeError(f"expected src of {x} to be bufferizable")
  return root.replace(src=(dest, buffer_src.view(unwrap(x.st))))

def ensure_realized(ctx:dict[UOp, UOp], root:UOp):
  pruned = dedup([x.base for x in root.src]) # TODO: is this worthy of its own rule?
  new_src = [x if (buf:=create_buffer(ctx, x.base)) is None else buf.view(x.st) for x in pruned]
  return None if tuple(new_src) == root.src else root.replace(src=tuple(new_src))

bufferize = PatternMatcher([
  # ensure COPY and BUFFER_VIEW inputs are realized
  (UPat({Ops.COPY, Ops.BUFFER_VIEW}, name="root", src=(UPat.var("dest"), UPat.var("x"),)), bufferize_input),
  # ensure SINKED uops are realized
  (UPat(Ops.SINK, name="root"), ensure_realized),
  # allocate new bufs for contiguous and copy
  (UPat({Ops.COPY, Ops.CONTIGUOUS}, name="root"), create_buffer),
  # simple rule for REDUCE_AXIS. TODO: fuse when it makes sense
  (UPat(Ops.REDUCE_AXIS, name="root"), create_buffer),
  # realize before expand or unsafe pad ops
  (UPat(Ops.VIEW, name="view", src=(UPat.var("src"),)), view_src),

  # add the pre existing buffers in EMPTY and ASSIGN
  (UPat(Ops.EMPTY, name="root", src=(UPat.var("target"),)), add_target_buf),
  (UPat(Ops.ASSIGN, name="root", src=(UPat.var("target"), UPat())), add_target_buf),

  # create subbuffer for BUFFER_VIEW
  (UPat(Ops.BUFFER_VIEW, name="root", src=(UPat(), UPat.var("src"))), create_subbuffer),
])

# ** deal with schedule variables

def unbind_st_vars(ctx:dict[Variable, int], root:UOp):
  st = unwrap(root.st).simplify()
  try:
    st, var_vals = st.unbind()
    ctx.update(var_vals)
  except AssertionError: pass # TODO: can unbind just return if the ShapeTracker has already been unbound?
  return root.replace(arg=st) if st != root.st else None

def unbind_var(ctx:dict[Variable, int], root:UOp):
  var, val = root.unbind()
  ctx[var.replace(src=())] = val
  return var

unbind_vars = PatternMatcher([
  (UPat(Ops.VIEW, name="root"), unbind_st_vars),
  (UPat(Ops.BIND, name="root"), unbind_var),
  # TODO: for some reason symbolic sts should keep existing on const reduce even if they are unmasked
  (UPat({Ops.CONST, Ops.DEFINE_VAR}, name="root", src=(UPat(),)), lambda root:root.replace(src=()) if all_int(root.shape) \
      else UOp(Ops.VALID, dtypes.bool, (unwrap(root.st).to_uop(),)).where(root.replace(src=()), 0)),
])

# ** deal with ImageDType

def can_image(root:UOp):
  assert root is root.base, f"can only make things that can't be images not images in base {root}"
  if not isinstance(dtype:=root.dtype, ImageDType) or root.st is None: return None
  if (prod(root.shape) != prod(dtype.shape) or not any(root.shape[x]%4 == 0 for x in unwrap(root.st).unit_stride_axes())):
    if DEBUG >= 2: print(f"forcing image {dtype} with shape {root.shape} to {dtype.base}")
    return root.replace(dtype=root.dtype.base)

image_pm = PatternMatcher([
  # sometimes we make things that can't be images not images (must be base)
  (UPat(set(Ops)-{Ops.VIEW}, name="root"), can_image),
])

# ** ast rewrite

def add_loads(ctx:list[UOp], buf:UOp):
  if buf not in ctx: ctx.append(buf)
  glbl = UOp(Ops.DEFINE_GLOBAL, buf.dtype.ptr(size=buf.size), (), ctx.index(buf))
  return UOp(Ops.LOAD, buf.dtype.base, (glbl, unwrap(buf.st).to_uop()))

def add_stores(sink:UOp):
  if all(x.op is Ops.STORE for x in sink.src): return None
  new_src: list[UOp] = []
  for i,x in enumerate(dedup(sink.src)):
    glbl = UOp(Ops.DEFINE_GLOBAL, x.dtype.ptr(size=x.size), (), i)
    new_src.append(UOp.store(glbl, ShapeTracker.from_shape(x.shape).to_uop(), x))
  return sink.replace(src=tuple(new_src))

debufferize = PatternMatcher([
  (UPat(Ops.BUFFER, name="buf"), add_loads),
  (UPat(Ops.SINK, name="sink"), add_stores),
])

view_right = PatternMatcher([
])

to_si = PatternMatcher([
  (UPat(Ops.SINK, src=(UPat.store(UPat(), UPat(), UPat(GroupOp.Meta, name="meta")))), lambda meta:meta),
  (UPat(Ops.CONTIGUOUS, src=(UPat.var("x"),)), lambda x:x),
  (UPat(Ops.ASSIGN, src=(UPat(), UPat.var("x"),)), lambda x:x),
  # in general once things are loaded they aren't image
  (UPat(set(Ops)-{Ops.DEFINE_GLOBAL}, name="root"), lambda root:root.replace(dtype=root.dtype.base) if isinstance(root.dtype, ImageDType) else None),
])

# ** schedule creation

@track_rewrites(named=True)
def create_schedule_with_vars(outs:list[UOp]) -> tuple[list[ScheduleItem], dict[Variable, int], dict[UOp, UOp]]:
  # verify
  sink = UOp.sink(*outs)
  type_verify(list(sink.toposort), tensor_uop_spec)

  # create kernels from the schedule graph
  realizes: dict[UOp, UOp] = {}
  tensor_map = graph_rewrite_map(sink, remove_movement_ops+sym+image_pm)
  buffer_map = graph_rewrite_map(tensor_map[sink], remove_movement_ops+sym+bufferize, realizes)

  # schedule
  schedule: list[ScheduleItem] = []
  var_vals: dict[Variable, int] = {}
  for k,v in realizes.items():
    ast = graph_rewrite(v.sink(), debufferize+view_left+remove_movement_ops, bufs:=[k])
    schedule.append(ScheduleItem(graph_rewrite(ast, unbind_vars+to_si, var_vals), tuple(b.buffer for b in bufs), ()))
    for b in bufs: b.buffer.ref(1)

  # update tensor refs
  rev_realize = {v:k for k,v in realizes.items()}
  becomes_map: dict[UOp, UOp] = {}
  for k,v in tensor_map.items():
    if (buf_ref:=buffer_map.get(v)) is None or k.base is buf_ref.base: continue
    if (realized:=rev_realize.get(buf_ref)) is None and buf_ref.base.op is Ops.BUFFER: realized = buf_ref
    if realized is not None: becomes_map[k] = realized.view(unwrap(k.st))

  # confirm everything was scheduled correctly
  allocated_bufs = set(realizes)
  backrefed_bufs = set([x.base for x in becomes_map.values()])
  if len(zombies:=(allocated_bufs - backrefed_bufs)) != 0:
    if DEBUG >= 3:
      for z in zombies: print(z.arg[0], rev_realize.get(z))
    raise AssertionError(f"have zombie bufs leftover {zombies}")
  return schedule, var_vals, becomes_map
