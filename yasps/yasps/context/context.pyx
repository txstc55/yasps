import pycuda.driver as cuda
from yasps.helper import timed

class context:
  __main_ctx = None
  __named_contexts = {}
  __current_ctx = None

  @classmethod
  def _ensure_init(cls):
    if cls.__main_ctx is None:
      cls.__main_ctx = cuda.Context.get_current()
      assert cls.__main_ctx is not None, "Must have an active CUDA context"
      cls.__current_ctx = cls.__main_ctx

  def __init__(self):
    context._ensure_init()

  @timed("context.useDefaultContext")
  def useDefaultContext(self):
    if context.__current_ctx == context.__main_ctx:
      return
    # first we pop the active context (if any) to avoid nesting contexts
    try:
      current_ctx = cuda.Context.get_current()
      if current_ctx is not None:
        current_ctx.pop()  # pop the current context to make it inactive
    except cuda.LogicError:
      pass  # no context was active, so we can ignore this error
    context.__main_ctx.push()  # push the main context to make it active
    context.__current_ctx = context.__main_ctx

  @timed("context.useNamedContext")
  def useNamedContext(self, name: str):
    if name not in self.__named_contexts:
      device = cuda.Device(0)
      current_ctx = cuda.Context.get_current()
      current_ctx.pop()
      context.__named_contexts[name] = device.make_context()
      context.__current_ctx = context.__named_contexts[name]
      return

    context.__current_ctx.pop()
    context.__named_contexts[name].push()
    context.__current_ctx = context.__named_contexts[name]
