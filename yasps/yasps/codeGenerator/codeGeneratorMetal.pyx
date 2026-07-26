"""Metal counterpart to ``codeGenerator.pyx`` for symbolic attributes.

This module deliberately does not evaluate an attribute graph with MLX array
operations. It emits one Metal program whose thread-local temporaries mirror
the CUDA/Eigen code generator, then lets MLX compile and cache that source.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable

import mlx.core as mx

import importlib

ya = importlib.import_module("yasps.attribute")
from yasps.backend import GPUArray


@dataclass(frozen=True)
class _Resource:
  kind: str
  name: str
  value: Callable[[], GPUArray]


@dataclass(frozen=True)
class _Module:
  key: int
  name: str
  resources: tuple[tuple[str, str], ...]
  dependencies: tuple[int, ...]
  source: str


_MODULE_CACHE: dict[int, _Module] = {}


def _metal_array(value):
  return value._array if isinstance(value, GPUArray) else value


def _float_literal(value: float) -> str:
  literal = repr(float(value))
  if "e" not in literal.lower() and "." not in literal:
    literal += ".0"
  return literal + "f"


def _operator_value(operator, pairs):
  for candidate, value in pairs:
    if operator == candidate:
      return value
  raise KeyError(operator.name)


def _resource_symbol(kind: str, name: str) -> str:
  digest = sha256(f"{kind}:{name}".encode()).hexdigest()[:16]
  return f"yasps_{kind}_{digest}"


class MetalProgram:
  """A generated, JIT-compiled Metal program for one symbolic attribute."""

  def __init__(self, attribute):
    if attribute.correspondance is None:
      raise ValueError("A computed Metal attribute needs a correspondence.")
    self.attribute = attribute
    self._resources: dict[tuple[str, str], _Resource] = {}
    self._variables = 0
    self._current_root = None
    self._current_dependencies: list[int] = []
    self._modules: dict[int, _Module] = {}
    self._collect(attribute, set())
    self.resources = sorted(
      self._resources.values(), key=lambda resource: (resource.kind, resource.name)
    )
    self._resource_names = {
      (resource.kind, resource.name): _resource_symbol(
        resource.kind, resource.name
      )
      for resource in self.resources
    }
    self._resource_indices = {
      (resource.kind, resource.name): index
      for index, resource in enumerate(self.resources)
    }
    self._float_resource_indices = tuple(
      index
      for index, resource in enumerate(self.resources)
      if resource.kind == "data"
    )
    self._uint_resource_indices = tuple(
      index
      for index, resource in enumerate(self.resources)
      if resource.kind != "data"
    )
    self._resource_categories = {
      (resource.kind, resource.name): (
        "yasps_float_resources"
        if resource.kind == "data"
        else "yasps_uint_resources"
      )
      for resource in self.resources
    }
    root_module = self._compile_module(attribute)
    self.root_module = root_module
    self.modules = self._ordered_modules(root_module.key)
    module_sources = [module.source for module in self.modules]
    self.header = "\n\n".join(
      [
        (
          Path(__import__("yasps").__file__).parent
          / "kernel"
          / "metalLinalg.metal"
        ).read_text(),
        *module_sources,
      ]
    )
    result_name = f"root_result_{root_module.key}".replace("-", "neg")
    lines: list[str] = [
      "const uint instance = thread_position_in_grid.x;",
      "if (instance >= max_index[0]) return;",
      f"float {result_name}[{attribute.size}];",
      f"{root_module.name}("
      f"{self._external_module_arguments(root_module, suffix=True)}"
      f"instance, {result_name});",
    ]
    lines.append(
      f"for (ushort i = 0; i < {attribute.size}; ++i) "
      f"result[instance * {attribute.size} + i] = {result_name}[i];"
    )
    self.source = "\n".join(lines)
    digest = sha256((self.header + self.source).encode()).hexdigest()[:16]
    self.name = f"yasps_attribute_{digest}"
    self.input_names = [*self.resource_input_names, "max_index"]
    self.kernel = mx.fast.metal_kernel(
      name=self.name,
      input_names=self.input_names,
      output_names=["result"],
      header=self.header,
      source=self.source,
      compile_options={"math_mode": "fast"},
    )

  def _add_resource(self, kind: str, owner, value: Callable[[], GPUArray]):
    name = owner.fullName
    self._resources.setdefault((kind, name), _Resource(kind, name, value))

  def _collect(self, attribute, seen: set[int]):
    identity = id(attribute)
    if identity in seen:
      return
    seen.add(identity)
    operator = attribute.operator
    if operator == ya.DATA or operator == ya.CONSTANT:
      self._add_resource("data", attribute, lambda item=attribute: item.value)
      return
    if operator in (ya.JOIN, ya.SUM, ya.AVERAGE):
      connection = attribute.through
      self._add_resource(
        "indices", connection, lambda item=connection: item.value
      )
      if connection.dimension == 0:
        self._add_resource(
          "csr", connection, lambda item=connection: item.compressedRows
        )
    if operator == ya.UNION:
      primitive_union = attribute.correspondance
      self._add_resource(
        "union",
        primitive_union,
        lambda item=primitive_union: item.children_primitive_counts_gpu,
      )
    for child in attribute.children:
      self._collect(child, seen)

  def _resource(self, kind: str, owner) -> str:
    return self._resource_names[(kind, owner.fullName)]

  def _resource_keys(self, attribute) -> tuple[tuple[str, str], ...]:
    keys: set[tuple[str, str]] = set()
    seen: set[int] = set()

    def collect(current):
      identity = id(current)
      if identity in seen:
        return
      seen.add(identity)
      operator = current.operator
      if operator == ya.DATA or operator == ya.CONSTANT:
        keys.add(("data", current.fullName))
        return
      if operator in (ya.JOIN, ya.SUM, ya.AVERAGE):
        connection = current.through
        keys.add(("indices", connection.fullName))
        if connection.dimension == 0:
          keys.add(("csr", connection.fullName))
      if operator == ya.UNION:
        keys.add(("union", current.correspondance.fullName))
      for child in current.children:
        collect(child)

    collect(attribute)
    return tuple(sorted(keys))

  def _module_arguments(self, module: _Module, suffix: bool = False) -> str:
    arguments = [self._resource_names[key] for key in module.resources]
    if suffix and arguments:
      return ", ".join(arguments) + ", "
    return ", ".join(arguments)

  def _external_module_arguments(
    self, module: _Module, suffix: bool = False
  ) -> str:
    arguments = [
      (
        f"{self._resource_categories[key]} + "
        f"yasps_resource_offsets[{self._resource_indices[key]}]"
      )
      for key in module.resources
    ]
    if suffix and arguments:
      return ", ".join(arguments) + ", "
    return ", ".join(arguments)

  @property
  def resource_input_names(self) -> list[str]:
    """Packed buffers keep generated kernels below Metal's binding limit."""

    names = []
    if self._float_resource_indices:
      names.append("yasps_float_resources")
    if self._uint_resource_indices:
      names.append("yasps_uint_resources")
    if self.resources:
      names.append("yasps_resource_offsets")
    return names

  def resource_arrays(self) -> list[mx.array]:
    """Pack current buffers on Metal and return them in ABI order."""

    arrays = [
      _metal_array(resource.value()).reshape((-1,))
      for resource in self.resources
    ]
    offsets = [0] * len(arrays)
    packed = []
    for indices, dtype in (
      (self._float_resource_indices, mx.float32),
      (self._uint_resource_indices, mx.uint32),
    ):
      if not indices:
        continue
      group = []
      offset = 0
      for index in indices:
        current = arrays[index]
        if current.dtype != dtype:
          current = current.astype(dtype)
        offsets[index] = offset
        offset += current.size
        group.append(current)
      packed.append(
        group[0] if len(group) == 1 else mx.concatenate(group, axis=0)
      )
    if self.resources:
      packed.append(mx.array(offsets, dtype=mx.uint32))
    return packed

  def root_call(self, instance: str, result: str) -> str:
    """Emit a call to the compiled root module from a surrounding kernel."""

    return (
      f"{self.root_module.name}("
      f"{self._external_module_arguments(self.root_module, suffix=True)}"
      f"{instance}, {result});"
    )

  def _compile_module(self, attribute) -> _Module:
    key = attribute.hash
    cached = _MODULE_CACHE.get(key)
    if cached is not None:
      self._modules[key] = cached
      for dependency in cached.dependencies:
        self._modules[dependency] = _MODULE_CACHE[dependency]
      return cached

    previous_root = self._current_root
    previous_variables = self._variables
    previous_dependencies = self._current_dependencies
    self._current_root = attribute
    self._variables = 0
    self._current_dependencies = []
    lines: list[str] = []
    output = self._emit(attribute, "instance_index", {}, lines, "  ")
    self._copy("result", output, attribute.size, lines, "  ")
    resource_keys = self._resource_keys(attribute)
    function_name = (
      "yasps_module_"
      + sha256(str(key).encode()).hexdigest()[:16]
    )
    parameters = []
    template_types = []
    for parameter_index, (kind, name) in enumerate(resource_keys):
      template_type = f"Resource{parameter_index}"
      template_types.append(f"typename {template_type}")
      parameters.append(
        f"{template_type} {_resource_symbol(kind, name)}"
      )
    parameters.extend(
      ["const uint instance_index", "thread float* result"]
    )
    template_header = (
      "template <" + ", ".join(template_types) + ">\n"
      if template_types
      else ""
    )
    source = (
      template_header
      + f"METAL_FUNC void {function_name}(\n  "
      + ",\n  ".join(parameters)
      + "\n) {\n"
      + "\n".join(lines)
      + "\n}"
    )
    module = _Module(
      key=key,
      name=function_name,
      resources=resource_keys,
      dependencies=tuple(dict.fromkeys(self._current_dependencies)),
      source=source,
    )
    _MODULE_CACHE[key] = module
    self._modules[key] = module
    output_directory = Path(".yasps_tmp/metal")
    output_directory.mkdir(parents=True, exist_ok=True)
    module_path = output_directory / f"{function_name}.metal"
    if not module_path.exists() or module_path.read_text() != source:
      module_path.write_text(source)
    self._current_root = previous_root
    self._variables = previous_variables
    self._current_dependencies = previous_dependencies
    return module

  def _ordered_modules(self, root_key: int) -> list[_Module]:
    order: list[_Module] = []
    seen: set[int] = set()

    def visit(key):
      if key in seen:
        return
      seen.add(key)
      module = self._modules.get(key, _MODULE_CACHE[key])
      for dependency in module.dependencies:
        visit(dependency)
      order.append(module)

    visit(root_key)
    return order

  def _emit_module_call(
    self, attribute, index, output, lines, indent
  ) -> str:
    module = self._compile_module(attribute)
    if module.key != self._current_root.hash:
      self._current_dependencies.append(module.key)
    arguments = self._module_arguments(module, suffix=True)
    lines.append(
      f"{indent}{module.name}({arguments}{index}, {output});"
    )
    return output

  def _new_variable(self, size: int, lines: list[str], indent: str) -> str:
    name = f"local_{self._variables}"
    self._variables += 1
    lines.append(f"{indent}float {name}[{size}];")
    return name

  def _copy(
    self,
    destination: str,
    source: str,
    size: int,
    lines: list[str],
    indent: str,
  ):
    lines.append(
      f"{indent}for (ushort i = 0; i < {size}; ++i) "
      f"{destination}[i] = {source}[i];"
    )

  def _emit(self, attribute, index, cache, lines, indent):
    key = attribute.hash
    if key in cache:
      return cache[key]

    operator = attribute.operator
    size = attribute.size
    output = self._new_variable(size, lines, indent)
    cache[key] = output
    if (
      attribute is not self._current_root
      and attribute.name != ""
      and attribute.generate_code
      and operator not in (ya.DATA, ya.CONSTANT)
    ):
      return self._emit_module_call(
        attribute, index, output, lines, indent
      )

    if operator == ya.FLOAT:
      lines.append(f"{indent}{output}[0] = {_float_literal(attribute.float_value)};")
      return output
    if operator == ya.INDEX:
      lines.append(f"{indent}{output}[0] = {attribute.index_value}.0f;")
      return output
    if operator == ya.DATA or operator == ya.CONSTANT:
      source = self._resource("data", attribute)
      correspondence = attribute.correspondance
      base = (
        "0"
        if correspondence.type in ("scene", "mesh")
        else f"({index}) * {size}"
      )
      lines.append(
        f"{indent}for (ushort i = 0; i < {size}; ++i) "
        f"{output}[i] = {source}[{base} + i];"
      )
      return output
    if operator == ya.ARRAY_ACCESS:
      source = self._emit(attribute.children[0], index, cache, lines, indent)
      position = attribute.children[1].index_value
      lines.append(f"{indent}{output}[0] = {source}[{position}];")
      return output
    if operator == ya.ARRAY:
      for position, child in enumerate(attribute.children):
        child_output = self._emit(child, index, cache, lines, indent)
        lines.append(f"{indent}{output}[{position}] = {child_output}[0];")
      return output
    if operator in (ya.ASCONSTANT, ya.INTERMEDIATE):
      source = self._emit(attribute.children[0], index, cache, lines, indent)
      self._copy(output, source, size, lines, indent)
      return output
    if operator == ya.NEG:
      source = self._emit(attribute.children[0], index, cache, lines, indent)
      lines.append(
        f"{indent}for (ushort i = 0; i < {size}; ++i) "
        f"{output}[i] = -{source}[i];"
      )
      return output
    if operator in (ya.SIN, ya.COS, ya.TAN, ya.COT, ya.ABS, ya.LOG, ya.SQRT):
      source = self._emit(attribute.children[0], index, cache, lines, indent)
      function = _operator_value(
        operator,
        (
          (ya.SIN, "sin"),
          (ya.COS, "cos"),
          (ya.TAN, "tan"),
          (ya.ABS, "abs"),
          (ya.LOG, "log"),
          (ya.SQRT, "sqrt"),
          (ya.COT, "tan"),
        ),
      )
      expression = (
        f"1.0f / metal::tan({source}[i])"
        if operator == ya.COT
        else f"metal::{function}({source}[i])"
      )
      lines.append(
        f"{indent}for (ushort i = 0; i < {size}; ++i) "
        f"{output}[i] = {expression};"
      )
      return output
    if operator in (ya.ADD, ya.SUB, ya.MUL, ya.DIV):
      left_attribute, right_attribute = attribute.children
      left = self._emit(left_attribute, index, cache, lines, indent)
      right = self._emit(right_attribute, index, cache, lines, indent)
      if operator == ya.MUL and left_attribute.size > 1 and right_attribute.size > 1:
        lines.append(
          f"{indent}for (ushort row = 0; row < {left_attribute.rows}; ++row) {{"
        )
        lines.append(
          f"{indent}  for (ushort column = 0; column < {right_attribute.cols}; ++column) {{"
        )
        lines.append(f"{indent}    float value = 0.0f;")
        lines.append(
          f"{indent}    for (ushort k = 0; k < {left_attribute.cols}; ++k) "
          f"value += {left}[row * {left_attribute.cols} + k] "
          f"* {right}[k * {right_attribute.cols} + column];"
        )
        lines.append(
          f"{indent}    {output}[row * {right_attribute.cols} + column] = value;"
        )
        lines.append(f"{indent}  }}")
        lines.append(f"{indent}}}")
      else:
        symbol = _operator_value(
          operator,
          ((ya.ADD, "+"), (ya.SUB, "-"), (ya.MUL, "*"), (ya.DIV, "/")),
        )
        left_index = "0" if left_attribute.size == 1 else "i"
        right_index = "0" if right_attribute.size == 1 else "i"
        lines.append(
          f"{indent}for (ushort i = 0; i < {size}; ++i) "
          f"{output}[i] = {left}[{left_index}] {symbol} {right}[{right_index}];"
        )
      return output
    if operator in (ya.BROADCAST_ADD, ya.BROADCAST_SUB):
      left = self._emit(attribute.children[0], index, cache, lines, indent)
      right = self._emit(attribute.children[1], index, cache, lines, indent)
      symbol = "+" if operator == ya.BROADCAST_ADD else "-"
      lines.append(
        f"{indent}for (ushort i = 0; i < {size}; ++i) "
        f"{output}[i] = {left}[i] {symbol} {right}[0];"
      )
      return output
    if operator in (ya.EQ, ya.NEQ, ya.LT, ya.LEQ, ya.GT, ya.GEQ):
      left = self._emit(attribute.children[0], index, cache, lines, indent)
      right = self._emit(attribute.children[1], index, cache, lines, indent)
      symbol = _operator_value(
        operator,
        (
          (ya.EQ, "=="),
          (ya.NEQ, "!="),
          (ya.LT, "<"),
          (ya.LEQ, "<="),
          (ya.GT, ">"),
          (ya.GEQ, ">="),
        ),
      )
      lines.append(f"{indent}{output}[0] = {left}[0] {symbol} {right}[0];")
      return output
    if operator in (ya.POW, ya.ATAN2):
      left = self._emit(attribute.children[0], index, cache, lines, indent)
      right = self._emit(attribute.children[1], index, cache, lines, indent)
      function = "pow" if operator == ya.POW else "atan2"
      lines.append(
        f"{indent}{output}[0] = metal::{function}({left}[0], {right}[0]);"
      )
      return output
    if operator == ya.SELECT:
      condition = self._emit(attribute.children[0], index, cache, lines, indent)
      when_true = self._emit(attribute.children[1], index, cache, lines, indent)
      when_false = self._emit(attribute.children[2], index, cache, lines, indent)
      lines.append(
        f"{indent}for (ushort i = 0; i < {size}; ++i) "
        f"{output}[i] = {condition}[0] != 0.0f ? "
        f"{when_true}[i] : {when_false}[i];"
      )
      return output
    if operator == ya.TRANSPOSE:
      source_attribute = attribute.children[0]
      source = self._emit(source_attribute, index, cache, lines, indent)
      lines.append(
        f"{indent}for (ushort row = 0; row < {attribute.rows}; ++row) "
        f"for (ushort column = 0; column < {attribute.cols}; ++column) "
        f"{output}[row * {attribute.cols} + column] = "
        f"{source}[column * {source_attribute.cols} + row];"
      )
      return output
    if operator == ya.ROW:
      source_attribute = attribute.children[0]
      source = self._emit(source_attribute, index, cache, lines, indent)
      row = attribute.children[1].index_value
      lines.append(
        f"{indent}for (ushort column = 0; column < {attribute.cols}; ++column) "
        f"{output}[column] = {source}[{row * source_attribute.cols} + column];"
      )
      return output
    if operator == ya.COL:
      source_attribute = attribute.children[0]
      source = self._emit(source_attribute, index, cache, lines, indent)
      column = attribute.children[1].index_value
      lines.append(
        f"{indent}for (ushort row = 0; row < {attribute.rows}; ++row) "
        f"{output}[row] = {source}[row * {source_attribute.cols} + {column}];"
      )
      return output
    if operator == ya.CROSS:
      left = self._emit(attribute.children[0], index, cache, lines, indent)
      right = self._emit(attribute.children[1], index, cache, lines, indent)
      lines.extend(
        [
          f"{indent}{output}[0] = {left}[1] * {right}[2] - {left}[2] * {right}[1];",
          f"{indent}{output}[1] = {left}[2] * {right}[0] - {left}[0] * {right}[2];",
          f"{indent}{output}[2] = {left}[0] * {right}[1] - {left}[1] * {right}[0];",
        ]
      )
      return output
    if operator == ya.NORM:
      source = self._emit(attribute.children[0], index, cache, lines, indent)
      lines.append(f"{indent}{output}[0] = 0.0f;")
      lines.append(
        f"{indent}for (ushort i = 0; i < {attribute.children[0].size}; ++i) "
        f"{output}[0] += {source}[i] * {source}[i];"
      )
      lines.append(f"{indent}{output}[0] = metal::sqrt({output}[0]);")
      return output
    if operator == ya.DET:
      source = self._emit(attribute.children[0], index, cache, lines, indent)
      lines.append(
        f"{indent}{output}[0] = "
        f"yasps_determinant<{attribute.children[0].rows}>({source});"
      )
      return output
    if operator == ya.INV:
      source = self._emit(attribute.children[0], index, cache, lines, indent)
      lines.append(
        f"{indent}yasps_inverse<{attribute.rows}>({source}, {output});"
      )
      return output
    if operator == ya.DOT:
      left = self._emit(attribute.children[0], index, cache, lines, indent)
      right = self._emit(attribute.children[1], index, cache, lines, indent)
      lines.append(f"{indent}{output}[0] = 0.0f;")
      lines.append(
        f"{indent}for (ushort i = 0; i < {attribute.children[0].size}; ++i) "
        f"{output}[0] += {left}[i] * {right}[i];"
      )
      return output
    if operator == ya.RESIZE:
      source = self._emit(attribute.children[0], index, cache, lines, indent)
      self._copy(output, source, size, lines, indent)
      return output
    if operator == ya.SPD:
      source = self._emit(attribute.children[0], index, cache, lines, indent)
      method = self._emit(attribute.children[1], index, cache, lines, indent)
      self._copy(output, source, size, lines, indent)
      lines.append(
        f"{indent}yasps_spd_project<{attribute.rows}>"
        f"({output}, int({method}[0]));"
      )
      return output
    if operator in (ya.JOIN, ya.SUM, ya.AVERAGE):
      return self._emit_connectivity(attribute, index, cache, lines, indent, output)
    if operator == ya.UNION:
      return self._emit_union(attribute, index, cache, lines, indent, output)
    raise NotImplementedError(
      f"Metal source generation does not support {operator.name}."
    )

  def _emit_connectivity(self, attribute, index, cache, lines, indent, output):
    connection = attribute.through
    child = attribute.children[0]
    indices = self._resource("indices", connection)
    if attribute.operator == ya.JOIN:
      lines.append(
        f"{indent}for (ushort connection_slot = 0; "
        f"connection_slot < {connection.dimension}; ++connection_slot) {{"
      )
      lines.append(
        f"{indent}  const uint connected_index = "
        f"{indices}[({index}) * {connection.dimension} + connection_slot];"
      )
      child_lines: list[str] = []
      if child.operator == ya.FLOAT or child.isFloatMat:
        child_output = self._emit(
          child, "connected_index", {}, child_lines, indent + "  "
        )
      else:
        child_output = self._new_variable(
          child.size, child_lines, indent + "  "
        )
        self._emit_module_call(
          child,
          "connected_index",
          child_output,
          child_lines,
          indent + "  ",
        )
      lines.extend(child_lines)
      lines.append(
        f"{indent}  for (ushort i = 0; i < {child.size}; ++i) "
        f"{output}[connection_slot * {child.size} + i] = {child_output}[i];"
      )
      lines.append(f"{indent}}}")
      return output

    csr = self._resource("csr", connection)
    lines.append(
      f"{indent}for (ushort i = 0; i < {attribute.size}; ++i) "
      f"{output}[i] = 0.0f;"
    )
    lines.append(f"{indent}const uint row_start = {csr}[{index}];")
    lines.append(f"{indent}const uint row_end = {csr}[({index}) + 1];")
    lines.append(
      f"{indent}for (uint connection_slot = row_start; "
      f"connection_slot < row_end; ++connection_slot) {{"
    )
    lines.append(
      f"{indent}  const uint connected_index = {indices}[connection_slot];"
    )
    child_lines = []
    if child.operator == ya.FLOAT or child.isFloatMat:
      child_output = self._emit(
        child, "connected_index", {}, child_lines, indent + "  "
      )
    else:
      child_output = self._new_variable(
        child.size, child_lines, indent + "  "
      )
      self._emit_module_call(
        child,
        "connected_index",
        child_output,
        child_lines,
        indent + "  ",
      )
    lines.extend(child_lines)
    lines.append(
      f"{indent}  for (ushort i = 0; i < {child.size}; ++i) "
      f"{output}[i] += {child_output}[i];"
    )
    lines.append(f"{indent}}}")
    if attribute.operator == ya.AVERAGE:
      lines.append(
        f"{indent}const float divisor = "
        f"float(metal::max(row_end - row_start, 1u));"
      )
      lines.append(
        f"{indent}for (ushort i = 0; i < {attribute.size}; ++i) "
        f"{output}[i] /= divisor;"
      )
    return output

  def _emit_union(self, attribute, index, cache, lines, indent, output):
    counts = self._resource("union", attribute.correspondance)
    lines.append(f"{indent}uint union_offset = 0;")
    for child_index, child in enumerate(attribute.children):
      lines.append(
        f"{indent}if (({index}) >= union_offset && ({index}) < "
        f"union_offset + {counts}[{child_index}]) {{"
      )
      child_lines: list[str] = []
      if child.operator == ya.FLOAT or child.isFloatMat:
        child_output = self._emit(
          child,
          f"({index}) - union_offset",
          {},
          child_lines,
          indent + "  ",
        )
      else:
        child_output = self._new_variable(
          child.size, child_lines, indent + "  "
        )
        self._emit_module_call(
          child,
          f"({index}) - union_offset",
          child_output,
          child_lines,
          indent + "  ",
        )
      lines.extend(child_lines)
      self._copy(output, child_output, attribute.size, lines, indent + "  ")
      lines.append(f"{indent}}}")
      if child_index + 1 < len(attribute.children):
        lines.append(f"{indent}union_offset += {counts}[{child_index}];")
    return output

  def run(self):
    count = self.attribute.correspondance.numInstances
    if count == 0:
      return mx.empty((0,), dtype=mx.float32)
    inputs = self.resource_arrays()
    inputs.append(mx.array([count], dtype=mx.uint32))
    return self.kernel(
      inputs=inputs,
      grid=(count, 1, 1),
      threadgroup=(min(count, 256), 1, 1),
      output_shapes=[(count * self.attribute.size,)],
      output_dtypes=[mx.float32],
    )[0]


class MetalGlobalKernel:
  """CUDA ``globalKernel`` equivalent backed by generated Metal source."""

  def __init__(self, attribute):
    self.program = MetalProgram(attribute)

  @property
  def kernelString(self):
    return self.program.header + "\n" + self.program.source

  def compute(self, output):
    output._array = self.program.run()
