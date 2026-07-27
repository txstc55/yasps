"""Mechanical CUDA/Eigen-to-MSL translation for generated YASPS functions."""

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


def translate_device_header(cuda_header: str) -> str:
  result = cuda_header.replace("__device__", "")
  result = re.sub(
    r"\bconst\s+double\s*\*",
    "device const float*",
    result,
  )
  result = re.sub(
    r"\bconst\s+unsigned\s+int\s*\*",
    "device const uint*",
    result,
  )
  result = re.sub(
    r"\bconst\s+unsigned\s+short\s+int\s*\*",
    "device const ushort*",
    result,
  )
  result = re.sub(r"\bdouble\s*\*", "thread float*", result)
  result = re.sub(r"\bdouble\b", "float", result)
  result = re.sub(r"\bunsigned short int\b", "ushort", result)
  result = re.sub(r"\bunsigned int\b", "uint", result)
  return _float_literals(result).strip()


def translate_device_kernel(
  cuda_kernel: str,
  cuda_header: str,
  rows: int,
  cols: int,
) -> tuple[str, str]:
  metal_header = translate_device_header(cuda_header)
  result = cuda_kernel.replace(cuda_header, metal_header, 1)
  result = result.replace("__device__", "")
  result = result.replace("__host__", "")
  result = _MATRIX_TYPE.sub(
    lambda match: (
      f"YaspsMatrix<{match.group(1)}, {match.group(2)}>"
    ),
    result,
  )
  result = re.sub(
    r"Eigen::Map<RowMat>\s+out\s*\(\s*result\s*\)\s*;",
    "RowMat out = {};",
    result,
  )
  result = _translate_matrix_maps(result)
  result = _translate_pointer_constructors(result)
  result = _translate_comma_initializers(result)
  result = result.replace("static_cast<unsigned int>", "uint")
  result = result.replace("unsigned short int", "ushort")
  result = result.replace("unsigned int", "uint")
  result = result.replace("double", "float")
  result = _float_literals(result)

  if rows * cols == 1:
    result = re.sub(
      r"result\[0\]\s*=\s*([^;]+);",
      r"result[0] = yasps_scalar_value(\1);",
      result,
    )
  if rows * cols > 1 and re.search(r"\bRowMat\s+out\b", result):
    closing = result.rfind("}")
    if closing < 0:
      raise ValueError("generated device kernel has no closing brace")
    store = (
      "\n  for (uint yasps_output_index = 0; "
      f"yasps_output_index < {rows * cols}; "
      "++yasps_output_index) {\n"
      "    result[yasps_output_index] = "
      "out.values[yasps_output_index];\n"
      "  }\n"
    )
    result = result[:closing] + store + result[closing:]
  return result, metal_header


def _translate_matrix_maps(source: str) -> str:
  pattern = re.compile(
    r"Eigen::Map<YaspsMatrix<(\d+),\s*(\d+)>>\s+"
    r"(\w+)\s*\(([^;]+)\)\s*;"
  )
  return pattern.sub(
    lambda match: (
      f"YaspsMatrix<{match.group(1)}, {match.group(2)}> "
      f"{match.group(3)} = "
      "yasps_matrix_from_pointer"
      f"<{match.group(1)}, {match.group(2)}>"
      f"({match.group(4)});"
    ),
    source,
  )


def _translate_pointer_constructors(source: str) -> str:
  pattern = re.compile(
    r"YaspsMatrix<(\d+),\s*(\d+)>\s+(\w+)\s*"
    r"\(\s*(\(?[A-Za-z_]\w*\)?\.data\(\))\s*\)\s*;"
  )
  return pattern.sub(
    lambda match: (
      f"YaspsMatrix<{match.group(1)}, {match.group(2)}> "
      f"{match.group(3)} = "
      "yasps_matrix_from_pointer"
      f"<{match.group(1)}, {match.group(2)}>"
      f"({match.group(4)});"
    ),
    source,
  )


def _translate_comma_initializers(source: str) -> str:
  marker = re.compile(r"\b([A-Za-z_]\w*)\s*<<")
  cursor = 0
  pieces: list[str] = []
  while True:
    match = marker.search(source, cursor)
    if match is None:
      pieces.append(source[cursor:])
      break
    pieces.append(source[cursor : match.start()])
    expression_start = match.end()
    expression_end = _find_statement_end(source, expression_start)
    expressions = _split_top_level_commas(
      source[expression_start:expression_end]
    )
    target = match.group(1)
    indentation_start = source.rfind("\n", 0, match.start()) + 1
    indentation = source[indentation_start : match.start()]
    assignments = "\n".join(
      f"{indentation}{target}.values[{index}] = {expression.strip()};"
      for index, expression in enumerate(expressions)
    )
    pieces.append(assignments.lstrip() if not pieces[-1].endswith("\n") else assignments)
    cursor = expression_end + 1
  return "".join(pieces)


def _find_statement_end(source: str, start: int) -> int:
  depth = 0
  for index in range(start, len(source)):
    character = source[index]
    if character in "([{":
      depth += 1
    elif character in ")]}":
      depth -= 1
    elif character == ";" and depth == 0:
      return index
  raise ValueError("unterminated generated comma initializer")


def _split_top_level_commas(source: str) -> list[str]:
  result: list[str] = []
  depth = 0
  start = 0
  for index, character in enumerate(source):
    if character in "([{":
      depth += 1
    elif character in ")]}":
      depth -= 1
    elif character == "," and depth == 0:
      result.append(source[start:index])
      start = index + 1
  result.append(source[start:])
  return result


def _float_literals(source: str) -> str:
  return _FLOAT_LITERAL.sub(lambda match: match.group(1) + "f", source)


__all__ = ["translate_device_header", "translate_device_kernel"]
