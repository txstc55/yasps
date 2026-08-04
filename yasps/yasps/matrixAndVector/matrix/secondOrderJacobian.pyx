from __future__ import annotations

from typing import List

from yasps.attribute import attribute
from yasps.matrix import matrix
from yasps.secondOrderJacobianIndicesKernel import secondOrderJacobianIndicesKernel


class secondOrderJacobian(matrix):
  """Rectangular matrix placeholder for a mixed second derivative."""

  def __init__(
    self,
    row_wrt: List[attribute],
    column_wrt: List[attribute]
  ):
    self.__row_wrt = self.__validateTargets(row_wrt, "row_wrt")
    self.__column_wrt = self.__validateTargets(column_wrt, "column_wrt")
    self.__row_start_indices = self.__computeStartIndices(self.__row_wrt)
    self.__column_start_indices = self.__computeStartIndices(self.__column_wrt)

    super().__init__(
      self.__row_start_indices[-1],
      self.__column_start_indices[-1]
    )

    self.__indices_kernels: List[secondOrderJacobianIndicesKernel] = []
    self.__sources: List[attribute] = []
    self.__second_order_jacobians: List[attribute] = []


    self.__indices_kernels_dynamic: List[secondOrderJacobianIndicesKernel] = []
    self.__sources_dynamic: List[attribute] = []
    self.__second_order_jacobians_dynamic: List[attribute] = []

  def __validateTargets(
    self,
    targets: List[attribute],
    name: str
  ) -> List[attribute]:
    if not isinstance(targets, list) or len(targets) == 0:
      raise ValueError(
        f"secondOrderJacobian.__init__: {name} must be a non-empty list."
      )

    result: List[attribute] = []
    seen = set()
    for target in targets:
      if not isinstance(target, attribute):
        raise TypeError(
          f"secondOrderJacobian.__init__: every {name} item must be an attribute."
        )
      if target.isDynamic:
        raise ValueError(
          f"secondOrderJacobian.__init__: {name} cannot contain dynamic attributes."
        )
      if target.hash in seen:
        raise ValueError(
          f"secondOrderJacobian.__init__: {name} contains a duplicate target."
        )
      seen.add(target.hash)
      result.append(target)
    return result

  def __computeStartIndices(self, targets: List[attribute]) -> List[int]:
    starts = [0]
    for target in targets:
      starts.append(
        starts[-1] + target.size * target.correspondance.numInstances
      )
    return starts

  @property
  def row_wrt(self) -> List[attribute]:
    return self.__row_wrt

  @property
  def column_wrt(self) -> List[attribute]:
    return self.__column_wrt

  @property
  def row_start_indices(self) -> List[int]:
    return self.__row_start_indices

  @property
  def column_start_indices(self) -> List[int]:
    return self.__column_start_indices

  @property
  def indices_kernels(self) -> List[secondOrderJacobianIndicesKernel]:
    return self.__indices_kernels

  @indices_kernels.setter
  def indices_kernels(
    self,
    value: List[secondOrderJacobianIndicesKernel]
  ) -> None:
    if not isinstance(value, list):
      raise TypeError("secondOrderJacobian.indices_kernels: value must be a list.")
    if any(not isinstance(item, secondOrderJacobianIndicesKernel) for item in value):
      raise TypeError(
        "secondOrderJacobian.indices_kernels: every item must be a secondOrderJacobianIndicesKernel."
      )
    self.__indices_kernels = list(value)

  @property
  def indices_kernels_dynamic(self) -> List[secondOrderJacobianIndicesKernel]:
    return self.__indices_kernels_dynamic

  @indices_kernels_dynamic.setter
  def indices_kernels_dynamic(
    self,
    value: List[secondOrderJacobianIndicesKernel]
  ) -> None:
    if not isinstance(value, list):
      raise TypeError(
        "secondOrderJacobian.indices_kernels_dynamic: value must be a list."
      )
    if any(not isinstance(item, secondOrderJacobianIndicesKernel) for item in value):
      raise TypeError(
        "secondOrderJacobian.indices_kernels_dynamic: every item must be a secondOrderJacobianIndicesKernel."
      )
    self.__indices_kernels_dynamic = list(value)

  @property
  def second_order_jacobians(self) -> List[attribute]:
    return self.__second_order_jacobians

  @second_order_jacobians.setter
  def second_order_jacobians(self, value: List[attribute]) -> None:
    if not isinstance(value, list):
      raise TypeError(
        "secondOrderJacobian.second_order_jacobians: value must be a list."
      )
    if any(not isinstance(item, attribute) for item in value):
      raise TypeError(
        "secondOrderJacobian.second_order_jacobians: every item must be an attribute."
      )
    self.__second_order_jacobians = list(value)

  @property
  def second_order_jacobians_dynamic(self) -> List[attribute]:
    return self.__second_order_jacobians_dynamic

  @second_order_jacobians_dynamic.setter
  def second_order_jacobians_dynamic(self, value: List[attribute]) -> None:
    if not isinstance(value, list):
      raise TypeError(
        "secondOrderJacobian.second_order_jacobians_dynamic: value must be a list."
      )
    if any(not isinstance(item, attribute) for item in value):
      raise TypeError(
        "secondOrderJacobian.second_order_jacobians_dynamic: every item must be an attribute."
      )
    self.__second_order_jacobians_dynamic = list(value)

  @property
  def sources(self) -> List[attribute]:
    return self.__sources

  @sources.setter
  def sources(self, value: List[attribute]) -> None:
    if not isinstance(value, list):
      raise TypeError("secondOrderJacobian.sources: value must be a list.")
    if any(not isinstance(item, attribute) for item in value):
      raise TypeError(
        "secondOrderJacobian.sources: every item must be an attribute."
      )
    self.__sources = list(value)

  @property
  def sources_dynamic(self) -> List[attribute]:
    return self.__sources_dynamic

  @sources_dynamic.setter
  def sources_dynamic(self, value: List[attribute]) -> None:
    if not isinstance(value, list):
      raise TypeError(
        "secondOrderJacobian.sources_dynamic: value must be a list."
      )
    if any(not isinstance(item, attribute) for item in value):
      raise TypeError(
        "secondOrderJacobian.sources_dynamic: every item must be an attribute."
      )
    self.__sources_dynamic = list(value)

  def __add__(self, other: secondOrderJacobian):
    if not isinstance(other, secondOrderJacobian):
      raise TypeError(f"secondOrderJacobian.__add__: unsupported operand type(s) for +: 'secondOrderJacobian' and '{type(other).__name__}'")
    if len(self.__row_wrt) != len(other.row_wrt):
      raise ValueError("secondOrderJacobian.__add__: row_wrt length mismatch.")
    for left, right in zip(self.__row_wrt, other.row_wrt):
      if left.hash != right.hash:
        raise ValueError("secondOrderJacobian.__add__: row_wrt mismatch.")
    if len(self.__column_wrt) != len(other.column_wrt):
      raise ValueError("secondOrderJacobian.__add__: column_wrt length mismatch.")
    for left, right in zip(self.__column_wrt, other.column_wrt):
      if left.hash != right.hash:
        raise ValueError("secondOrderJacobian.__add__: column_wrt mismatch.")
    result = secondOrderJacobian(
      self.__row_wrt,
      self.__column_wrt
    )
    result.indices_kernels = self.__indices_kernels + other.indices_kernels
    result.sources = self.__sources + other.sources
    result.second_order_jacobians = self.__second_order_jacobians + other.second_order_jacobians
    result.sources_dynamic = self.__sources_dynamic + other.sources_dynamic
    result.indices_kernels_dynamic = self.__indices_kernels_dynamic + other.indices_kernels_dynamic
    result.second_order_jacobians_dynamic = self.__second_order_jacobians_dynamic + other.second_order_jacobians_dynamic
    return result
