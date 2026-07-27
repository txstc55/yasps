"""Translate generated CUDA Hessian entry points into Metal kernels."""

from __future__ import annotations

import re


_MATRIX_TYPE = re.compile(
  r"Eigen::Matrix<double,\s*(\d+),\s*(\d+)"
  r"(?:,\s*Eigen::RowMajor)?\s*>"
)
_FLOAT_LITERAL = re.compile(
  r"(?<![\w.])("
  r"(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?"
  r"|\d+[eE][+-]?\d+"
  r")(?![\w.]|f)"
)
_ATOMIC_OUTPUTS = {
  "gradient",
  "hessian_blocks",
  "diagonal",
  "diagonal_blocks",
}


def translate_hessian_kernel(
  cuda_source: str,
  function_name: str,
  header_include: str | None = "yasps_hessian_headers.metal",
  collapse_groups: bool = False,
  max_threads_per_threadgroup: int | None = None,
) -> str:
  """Lower one generated CUDA global Hessian function to MSL."""

  marker = f"__global__ void {function_name}"
  function_start = cuda_source.find(marker)
  if function_start < 0:
    raise ValueError(f"CUDA Hessian function {function_name!r} is missing")
  parameters_start = cuda_source.find("(", function_start) + 1
  parameters_end = _matching_delimiter(
    cuda_source,
    parameters_start - 1,
    "(",
    ")",
  )
  body_start = cuda_source.find("{", parameters_end)
  body_end = _matching_delimiter(cuda_source, body_start, "{", "}")

  preamble = cuda_source[:function_start]
  preamble = re.sub(
    r"^\s*#include[^\n]*\n",
    "",
    preamble,
    flags=re.MULTILINE,
  )
  preamble = preamble.replace('extern "C"{', "")
  preamble = _translate_common(preamble).strip()

  raw_parameters = cuda_source[parameters_start:parameters_end]
  raw_parameters = re.sub(r"//[^\n]*", "", raw_parameters)
  declarations = [
    item.strip()
    for item in _split_top_level(raw_parameters)
    if item.strip()
  ]
  if collapse_groups:
    nth_index = next(
      index
      for index, declaration in enumerate(declarations)
      if declaration.endswith("nth_gradient_size")
    )
    declarations.insert(
      nth_index + 1,
      "const unsigned int total_instance_count",
    )

  fields = []
  aliases = []
  for index, declaration in enumerate(declarations):
    field_type, name = _translate_parameter(declaration)
    fields.append(f"  {field_type} {name} [[id({index})]];")
    aliases.append(f"  {field_type} {name} = arguments.{name};")

  body = cuda_source[body_start + 1:body_end]
  if collapse_groups:
    body, replacements = re.subn(
      (
        r"const\s+unsigned\s+int\s+start\s*=\s*"
        r"groupedIndicesOuter\[nth_gradient_size\]\s*;\s*"
        r"const\s+unsigned\s+int\s+end\s*=\s*"
        r"groupedIndicesOuter\[nth_gradient_size\s*\+\s*1\]\s*;"
      ),
      (
        "const unsigned int start = 0;\n"
        "  const unsigned int end = total_instance_count;"
      ),
      body,
      count=1,
    )
    if replacements != 1:
      raise ValueError(
        "CUDA Hessian kernel does not contain the expected "
        "gradient-size dispatch bounds"
      )
  body = re.sub(
    r"unsigned\s+int\s+([A-Za-z_]\w*)\s*=\s*"
    r"blockIdx\.x\s*\*\s*blockDim\.x\s*\+\s*threadIdx\.x\s*;",
    r"uint \1 = yasps_grid_index;",
    body,
  )
  body = _translate_common(body)

  argument_struct = f"{function_name}_arguments"
  pieces = [
    "#include <metal_stdlib>",
    "using namespace metal;",
    "",
  ]
  if header_include is not None:
    pieces.extend([f'#include "{header_include}"', ""])
  pieces.extend([
    (
      "inline void atomicAdd("
      "device atomic_float* target, float value) {\n"
      "  atomic_fetch_add_explicit(\n"
      "    target,\n"
      "    value,\n"
      "    memory_order_relaxed\n"
      "  );\n"
      "}"
    ),
    "",
  ])
  if preamble:
    pieces.extend([preamble, ""])
  pieces.extend([
    f"struct {argument_struct} {{",
    "\n".join(fields),
    "};",
    "",
  ])
  if max_threads_per_threadgroup is not None:
    pieces.append(
      "[[max_total_threads_per_threadgroup("
      f"{max_threads_per_threadgroup})]]"
    )
  pieces.extend([
    f"kernel void {function_name}(",
    (
      f"  device const {argument_struct}& arguments "
      "[[buffer(0)]],"
    ),
    "  uint yasps_grid_index [[thread_position_in_grid]]",
    ") {",
    "\n".join(aliases),
    body,
    "}",
    "",
  ])
  return "\n".join(pieces)


def _translate_parameter(declaration: str) -> tuple[str, str]:
  declaration = " ".join(declaration.split())
  name_match = re.search(r"([A-Za-z_]\w*)\s*$", declaration)
  if name_match is None:
    raise ValueError(f"cannot parse Hessian parameter {declaration!r}")
  name = name_match.group(1)
  source_type = declaration[:name_match.start()].strip()

  if "*" not in source_type:
    translated = _translate_scalar_type(
      source_type.removeprefix("const ").strip()
    )
    return translated, name

  is_const = source_type.startswith("const ")
  base = source_type.replace("const ", "", 1).replace("*", "").strip()
  translated_base = _translate_scalar_type(base)
  if name in _ATOMIC_OUTPUTS:
    return "device atomic_float*", name
  qualifier = "device const" if is_const else "device"
  return f"{qualifier} {translated_base}*", name


def _translate_scalar_type(source_type: str) -> str:
  replacements = {
    "double": "float",
    "unsigned int": "uint",
    "unsigned short int": "ushort",
    "short unsigned int": "ushort",
    "short int": "short",
  }
  return replacements.get(source_type, source_type)


def _translate_common(source: str) -> str:
  result = source.replace("__device__ __constant__", "constant")
  result = result.replace("__device__", "")
  result = result.replace("__host__", "")
  result = _MATRIX_TYPE.sub(
    lambda match: (
      f"YaspsMatrix<{match.group(1)}, {match.group(2)}>"
    ),
    result,
  )
  result = re.sub(
    r"(YaspsMatrix<\d+,\s*\d+>)::Zero\(\)",
    "{}",
    result,
  )
  result = result.replace("unsigned short int", "ushort")
  result = result.replace("short unsigned int", "ushort")
  result = result.replace("unsigned int", "uint")
  result = result.replace("short int", "short")
  result = result.replace("double", "float")
  result = result.replace("static_cast<uint>", "uint")
  result = re.sub(
    r"\bprintf\s*\([^;]*\)\s*;",
    "",
    result,
    flags=re.DOTALL,
  )
  return _FLOAT_LITERAL.sub(
    lambda match: match.group(1) + "f",
    result,
  )


def _matching_delimiter(
  source: str,
  start: int,
  opening: str,
  closing: str,
) -> int:
  depth = 0
  for index in range(start, len(source)):
    character = source[index]
    if character == opening:
      depth += 1
    elif character == closing:
      depth -= 1
      if depth == 0:
        return index
  raise ValueError(f"unterminated {opening}{closing} section")


def _split_top_level(source: str) -> list[str]:
  result = []
  depth = 0
  start = 0
  for index, character in enumerate(source):
    if character in "([{<":
      depth += 1
    elif character in ")]}>":
      depth -= 1
    elif character == "," and depth == 0:
      result.append(source[start:index])
      start = index + 1
  result.append(source[start:])
  return result


__all__ = ["translate_hessian_kernel"]
