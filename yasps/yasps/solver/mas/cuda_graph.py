"""Small CUDA-runtime graph wrapper used by the device PCG hot path.

PyCUDA does not currently expose stream capture in its public driver module,
but its streams and kernels interoperate with CUDA Runtime API graph calls.  A
captured PCG iteration removes the Python-to-driver transition for each of its
individual kernels while keeping all numerical state in the existing PyCUDA
allocations.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass


class _ConditionalNodeParams(ctypes.Structure):
  _fields_ = [
    ("handle", ctypes.c_ulonglong),
    ("type", ctypes.c_int),
    ("size", ctypes.c_uint),
    ("phGraph_out", ctypes.POINTER(ctypes.c_void_p)),
  ]


class _GraphNodePayload(ctypes.Union):
  _fields_ = [
    ("reserved", ctypes.c_longlong * 29),
    ("conditional", _ConditionalNodeParams),
  ]


class _GraphNodeParams(ctypes.Structure):
  _anonymous_ = ("payload",)
  _fields_ = [
    ("type", ctypes.c_int),
    ("reserved0", ctypes.c_int * 3),
    ("payload", _GraphNodePayload),
    ("reserved2", ctypes.c_longlong),
  ]


class CUDAGraphError(RuntimeError):
  pass


class _RuntimeAPI:
  def __init__(self):
    runtime = ctypes.CDLL("libcudart.so")
    runtime.cudaGetErrorString.argtypes = [ctypes.c_int]
    runtime.cudaGetErrorString.restype = ctypes.c_char_p
    runtime.cudaStreamBeginCapture.argtypes = [ctypes.c_void_p, ctypes.c_int]
    runtime.cudaStreamBeginCapture.restype = ctypes.c_int
    runtime.cudaStreamEndCapture.argtypes = [
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_void_p),
    ]
    runtime.cudaStreamEndCapture.restype = ctypes.c_int
    runtime.cudaGraphInstantiateWithFlags.argtypes = [
      ctypes.POINTER(ctypes.c_void_p),
      ctypes.c_void_p,
      ctypes.c_ulonglong,
    ]
    runtime.cudaGraphInstantiateWithFlags.restype = ctypes.c_int
    runtime.cudaGraphCreate.argtypes = [
      ctypes.POINTER(ctypes.c_void_p),
      ctypes.c_uint,
    ]
    runtime.cudaGraphCreate.restype = ctypes.c_int
    runtime.cudaGraphAddChildGraphNode.argtypes = [
      ctypes.POINTER(ctypes.c_void_p),
      ctypes.c_void_p,
      ctypes.POINTER(ctypes.c_void_p),
      ctypes.c_size_t,
      ctypes.c_void_p,
    ]
    runtime.cudaGraphAddChildGraphNode.restype = ctypes.c_int
    runtime.cudaGraphConditionalHandleCreate.argtypes = [
      ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_void_p,
      ctypes.c_uint, ctypes.c_uint,
    ]
    runtime.cudaGraphConditionalHandleCreate.restype = ctypes.c_int
    runtime.cudaGraphAddNode.argtypes = [
      ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
      ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
      ctypes.POINTER(_GraphNodeParams),
    ]
    runtime.cudaGraphAddNode.restype = ctypes.c_int
    runtime.cudaGraphLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    runtime.cudaGraphLaunch.restype = ctypes.c_int
    runtime.cudaGraphExecDestroy.argtypes = [ctypes.c_void_p]
    runtime.cudaGraphExecDestroy.restype = ctypes.c_int
    runtime.cudaGraphDestroy.argtypes = [ctypes.c_void_p]
    runtime.cudaGraphDestroy.restype = ctypes.c_int
    self.runtime = runtime

  def check(self, status: int, operation: str) -> None:
    if status == 0:
      return
    detail = self.runtime.cudaGetErrorString(status)
    message = detail.decode("utf8", errors="replace") if detail else "unknown error"
    raise CUDAGraphError(f"{operation} failed: {message} ({status})")


_RUNTIME_API: _RuntimeAPI | None = None


def _runtime_api() -> _RuntimeAPI:
  global _RUNTIME_API
  if _RUNTIME_API is None:
    _RUNTIME_API = _RuntimeAPI()
  return _RUNTIME_API


@dataclass
class CapturedGraph:
  """An instantiated graph bound to the stream used for capture."""

  graph: ctypes.c_void_p
  executable: ctypes.c_void_p
  stream_handle: int

  @classmethod
  def capture(cls, stream, submit) -> "CapturedGraph":
    api = _runtime_api()
    stream_handle = int(stream.handle)
    stream_pointer = ctypes.c_void_p(stream_handle)
    api.check(
      api.runtime.cudaStreamBeginCapture(stream_pointer, 0),
      "cudaStreamBeginCapture",
    )
    try:
      submit()
    except BaseException:
      # Ending an invalidated capture restores the stream to a usable
      # state. Ignore that secondary status and preserve the root error.
      abandoned = ctypes.c_void_p()
      api.runtime.cudaStreamEndCapture(stream_pointer, ctypes.byref(abandoned))
      if abandoned.value:
        api.runtime.cudaGraphDestroy(abandoned)
      raise
    graph = ctypes.c_void_p()
    api.check(
      api.runtime.cudaStreamEndCapture(stream_pointer, ctypes.byref(graph)),
      "cudaStreamEndCapture",
    )
    executable = ctypes.c_void_p()
    try:
      api.check(
        api.runtime.cudaGraphInstantiateWithFlags(
          ctypes.byref(executable), graph, 0
        ),
        "cudaGraphInstantiateWithFlags",
      )
    except BaseException:
      api.runtime.cudaGraphDestroy(graph)
      raise
    return cls(graph, executable, stream_handle)

  @classmethod
  def repeat(cls, child: "CapturedGraph", count: int) -> "CapturedGraph":
    """Instantiate a chain containing ``count`` copies of a child graph."""
    if count <= 0:
      raise ValueError("CUDA graph repeat count must be positive")
    api = _runtime_api()
    graph = ctypes.c_void_p()
    api.check(api.runtime.cudaGraphCreate(ctypes.byref(graph), 0), "cudaGraphCreate")
    previous = ctypes.c_void_p()
    try:
      for _ in range(count):
        node = ctypes.c_void_p()
        dependencies = (
          ctypes.pointer(previous) if previous.value else None
        )
        api.check(
          api.runtime.cudaGraphAddChildGraphNode(
            ctypes.byref(node), graph, dependencies,
            1 if previous.value else 0, child.graph,
          ),
          "cudaGraphAddChildGraphNode",
        )
        previous = node
      executable = ctypes.c_void_p()
      api.check(
        api.runtime.cudaGraphInstantiateWithFlags(
          ctypes.byref(executable), graph, 0
        ),
        "cudaGraphInstantiateWithFlags",
      )
    except BaseException:
      api.runtime.cudaGraphDestroy(graph)
      raise
    return cls(graph, executable, child.stream_handle)

  @classmethod
  def conditional_while(cls, child: "CapturedGraph", capture_condition) -> "CapturedGraph":
    """Build a device-controlled while loop around ``child``.

    ``capture_condition`` receives the CUDA conditional handle and returns
    a captured graph whose final device kernel updates that handle. The
    loop has no host-side convergence polling or per-iteration launches.
    """
    api = _runtime_api()
    graph = ctypes.c_void_p()
    api.check(api.runtime.cudaGraphCreate(ctypes.byref(graph), 0), "cudaGraphCreate")
    condition_graph = None
    try:
      handle = ctypes.c_ulonglong()
      # Reinitialize the condition to true on every graph launch.
      api.check(
        api.runtime.cudaGraphConditionalHandleCreate(
          ctypes.byref(handle), graph, 1, 1
        ),
        "cudaGraphConditionalHandleCreate",
      )
      condition_graph = capture_condition(int(handle.value))
      if condition_graph.stream_handle != child.stream_handle:
        raise ValueError("conditional CUDA graphs must use the same stream")

      parameters = _GraphNodeParams()
      parameters.type = 0x0D  # cudaGraphNodeTypeConditional
      parameters.conditional.handle = handle.value
      parameters.conditional.type = 1  # cudaGraphCondTypeWhile
      parameters.conditional.size = 1
      parameters.conditional.phGraph_out = None
      conditional_node = ctypes.c_void_p()
      api.check(
        api.runtime.cudaGraphAddNode(
          ctypes.byref(conditional_node), graph, None, None, 0,
          ctypes.byref(parameters),
        ),
        "cudaGraphAddNode(while)",
      )
      body_graphs = parameters.conditional.phGraph_out
      if not body_graphs:
        raise CUDAGraphError("CUDA did not return a conditional body graph")
      body = body_graphs[0]
      iteration_node = ctypes.c_void_p()
      api.check(
        api.runtime.cudaGraphAddChildGraphNode(
          ctypes.byref(iteration_node), body, None, 0, child.graph
        ),
        "cudaGraphAddChildGraphNode(iteration)",
      )
      condition_node = ctypes.c_void_p()
      dependency = ctypes.pointer(iteration_node)
      api.check(
        api.runtime.cudaGraphAddChildGraphNode(
          ctypes.byref(condition_node), body, dependency, 1,
          condition_graph.graph,
        ),
        "cudaGraphAddChildGraphNode(condition)",
      )
      executable = ctypes.c_void_p()
      api.check(
        api.runtime.cudaGraphInstantiateWithFlags(
          ctypes.byref(executable), graph, 0
        ),
        "cudaGraphInstantiateWithFlags(while)",
      )
    except BaseException:
      api.runtime.cudaGraphDestroy(graph)
      raise
    finally:
      if condition_graph is not None:
        condition_graph.close()
    return cls(graph, executable, child.stream_handle)

  def launch(self) -> None:
    api = _runtime_api()
    api.check(
      api.runtime.cudaGraphLaunch(
        self.executable, ctypes.c_void_p(self.stream_handle)
      ),
      "cudaGraphLaunch",
    )

  def launch_many(self, count: int) -> None:
    api = _runtime_api()
    stream = ctypes.c_void_p(self.stream_handle)
    launch = api.runtime.cudaGraphLaunch
    for _ in range(count):
      api.check(launch(self.executable, stream), "cudaGraphLaunch")

  def close(self) -> None:
    api = _runtime_api()
    if self.executable.value:
      api.runtime.cudaGraphExecDestroy(self.executable)
      self.executable = ctypes.c_void_p()
    if self.graph.value:
      api.runtime.cudaGraphDestroy(self.graph)
      self.graph = ctypes.c_void_p()

  def __del__(self):  # pragma: no cover - context shutdown is process dependent
    try:
      self.close()
    except BaseException:
      pass
