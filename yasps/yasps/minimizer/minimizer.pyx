# cython: language_level=3
from __future__ import annotations
import time
from typing import List, Set, Optional, Tuple

import pycuda.autoinit
import numpy as np
import pycuda.gpuarray as gpuarray

from yasps.attribute import attribute, DATA
from yasps.differentiator import differentiator
from yasps.gradient import gradient
from yasps.hessian import hessian
from yasps.solverKernel import solverKernel
from yasps.diagonalBlockInverseKernel import diagonalBlockInverseKernel
from yasps.helper import timed


class energyRequest:
  """
  Stores one symbolic differentiation request until wrt is known.
  """

  def __init__(
    self,
    energy: attribute,
    targets: List[attribute] = [],
    projection_method = 1,
    save_intermediate = False,
    gradient_only = False,
    dynamic_instances = False,
    separate_hessian_jacobian = False
  ):
    self.__energy: attribute = energy
    self.__targets: List[attribute] = list(targets)
    self.__projection_method: int = projection_method
    self.__save_intermediate: bool = save_intermediate
    self.__gradient_only: bool = gradient_only
    self.__dynamic_instances: bool = dynamic_instances
    self.__separate_hessian_jacobian: bool = separate_hessian_jacobian

  @property
  def energy(self) -> attribute:
    return self.__energy

  @property
  def targets(self) -> List[attribute]:
    return self.__targets

  @property
  def projection_method(self) -> int:
    return self.__projection_method

  @property
  def save_intermediate(self) -> bool:
    return self.__save_intermediate

  @property
  def gradient_only(self) -> bool:
    return self.__gradient_only

  @property
  def dynamic_instances(self) -> bool:
    return self.__dynamic_instances

  @property
  def separate_hessian_jacobian(self) -> bool:
    return self.__separate_hessian_jacobian

  @property
  def hash(self) -> int:
    return self.__energy.hash


class minimizer:
  def __init__(self):
    # We first store energy requests symbolically, then differentiate them after wrt is known.
    self.__energy_requests: List[energyRequest] = []
    self.__energy_requests_dynamic: List[energyRequest] = []
    self.__differentiated_hessians: List[hessian] = []
    self.__differentiated_hessians_dynamic: List[hessian] = []
    self.__active_hessian: Optional[hessian] = None
    self.__active_hessian_ignore_hashes: Tuple[int, ...] = tuple()
    self.__needs_differentiation: bool = True

    self.__wrt: List[attribute] = []
    self.__gradient_object: Optional[gradient] = None
    self.__seen_pre_targets_full_names: Set[str] = set()

    self.__solver: Optional[solverKernel] = None
    self.__d_p1_b: gpuarray.GPUArray = gpuarray.empty(0, dtype=np.float64)
    self.__d_r: gpuarray.GPUArray = gpuarray.empty(0, dtype=np.float64)
    self.__d_c: gpuarray.GPUArray = gpuarray.empty(0, dtype=np.float64)
    self.__d_q: gpuarray.GPUArray = gpuarray.empty(0, dtype=np.float64)
    self.__d_s: gpuarray.GPUArray = gpuarray.empty(0, dtype=np.float64)
    self.__solution: gpuarray.GPUArray = gpuarray.empty(0, dtype=np.float64)
    self.__solutionSegments: List[gpuarray.GPUArray] = []
    self.__last5solutions: gpuarray.GPUArray = gpuarray.empty(0, dtype=np.float64)

    self.__diagonalBlockInverseKernel: Optional[diagonalBlockInverseKernel] = None
    self.__ignoredEnergyHashList: List[int] = []

  @property
  def solutionSegments(self) -> List[gpuarray.GPUArray]:
    return self.__solutionSegments

  @property
  def gradient(self) -> gpuarray.GPUArray:
    if self.__gradient_object is None:
      return gpuarray.empty(0, dtype=np.float64)
    return self.__gradient_object.value

  @property
  def gradientSegments(self) -> List[gpuarray.GPUArray]:
    if self.__gradient_object is None:
      return []
    return self.__gradient_object.gradient_segments

  @property
  def energies(self) -> List[energyRequest]:
    return self.__energy_requests

  @property
  def energiesDynamic(self) -> List[energyRequest]:
    return self.__energy_requests_dynamic

  @property
  def wrt(self) -> List[attribute]:
    return self.__wrt

  @property
  def diagonal(self) -> gpuarray.GPUArray:
    if self.__active_hessian is None:
      return gpuarray.empty(0, dtype=np.float64)
    return self.__active_hessian.diagonal

  def __markDifferentiationDirty(self) -> None:
    self.__needs_differentiation = True
    self.__active_hessian = None
    self.__active_hessian_ignore_hashes = tuple()

  def __resetSolutionBuffers(self) -> None:
    self.__solver = None
    self.__d_p1_b = gpuarray.empty(0, dtype=np.float64)
    self.__d_r = gpuarray.empty(0, dtype=np.float64)
    self.__d_c = gpuarray.empty(0, dtype=np.float64)
    self.__d_q = gpuarray.empty(0, dtype=np.float64)
    self.__d_s = gpuarray.empty(0, dtype=np.float64)
    self.__solution = gpuarray.empty(0, dtype=np.float64)
    self.__solutionSegments = []
    self.__last5solutions = gpuarray.empty(0, dtype=np.float64)
    self.__diagonalBlockInverseKernel = None

  def __initializeGradientLayout(self) -> None:
    self.__gradient_object = gradient(self.__wrt)
    self.__resetSolutionBuffers()

  def addEnergies(self, energies: List[attribute]) -> None:
    for item in energies:
      self.addEnergy(item)

  def addEnergy(self, e: attribute, targets: List[attribute] = [], projection_method = 1, save_intermediate = False, gradient_only = False, dynamic_instances = False, separate_hessian_jacobian = False) -> None:
    if e.name == "":
      raise ValueError("minimizer.addEnergy: energy attribute must have a name.")
    if gradient_only:
      raise NotImplementedError("minimizer.addEnergy: gradient_only is not supported in the Hessian-based minimizer yet.")

    for t in targets:
      self.__seen_pre_targets_full_names.add(t.fullName)

    new_request = energyRequest(
      e,
      targets=targets,
      projection_method=projection_method,
      save_intermediate=save_intermediate,
      gradient_only=gradient_only,
      dynamic_instances=dynamic_instances,
      separate_hessian_jacobian=separate_hessian_jacobian
    )
    existing_hashes = [item.hash for item in self.__energy_requests + self.__energy_requests_dynamic]
    if new_request.hash in existing_hashes:
      raise ValueError("minimizer.addEnergy: energy already exists.")

    if not dynamic_instances:
      self.__energy_requests.append(new_request)
    else:
      self.__energy_requests_dynamic.append(new_request)
    self.__markDifferentiationDirty()

  def addWrt(self, wrt: List[attribute]) -> None:
    seen_attribute_hashes: Set[int] = set()
    for att in wrt:
      if att.hash in seen_attribute_hashes:
        raise ValueError(f"minimizer.addWrt: wrt {att.fullName} is duplicate attribute.")
      if att.operator is not DATA:
        raise ValueError(f"minimizer.addWrt: wrt {att.fullName} is non-data attribute.")
      if att.isDynamic:
        raise ValueError(f"minimizer.addWrt: wrt {att.fullName} is dynamic attribute.")
      seen_attribute_hashes.add(att.hash)

    target_full_name_set = set([t.fullName for t in wrt])
    if not self.__seen_pre_targets_full_names.issubset(target_full_name_set):
      missing = self.__seen_pre_targets_full_names - target_full_name_set
      raise ValueError(f"minimizer.addWrt: target is missing attributes {missing} that are required by the energies added.")

    self.__wrt = list(wrt)
    self.__initializeGradientLayout()
    self.__markDifferentiationDirty()
    # self.__differentiateRequests()

  def __differentiateRequests(self) -> None:
    if not self.__needs_differentiation:
      return
    if len(self.__wrt) == 0:
      return

    differentiator_local = differentiator()
    self.__differentiated_hessians = []
    self.__differentiated_hessians_dynamic = []

    for request in self.__energy_requests:
      self.__differentiated_hessians.append(
        differentiator_local.diff2(
          [request.energy],
          self.__wrt,
          self.__wrt,
          local_targets=request.targets,
          projection_method=request.projection_method,
          save_intermediate=request.save_intermediate,
          separate_hessian_jacobian=request.separate_hessian_jacobian,
          dynamic_instances=False
        )
      )

    for request in self.__energy_requests_dynamic:
      self.__differentiated_hessians_dynamic.append(
        differentiator_local.diff2(
          [request.energy],
          self.__wrt,
          self.__wrt,
          local_targets=request.targets,
          projection_method=request.projection_method,
          save_intermediate=request.save_intermediate,
          separate_hessian_jacobian=request.separate_hessian_jacobian,
          dynamic_instances=True
        )
      )

    self.__needs_differentiation = False
    self.__active_hessian = None
    self.__active_hessian_ignore_hashes = tuple()

  @timed("minimizer.generateHessianAndGradient")
  def generateHessianAndGradient(self):
    if not self.__needs_differentiation:
      return
    start = time.time()
    self.__differentiateRequests()
    end = time.time()
    print(f"Autodiff computation: {1000.0 * (end - start)} ms")

  def __getActiveHessian(self) -> Optional[hessian]:
    self.__differentiateRequests()
    ignored_hashes = tuple(sorted(self.__ignoredEnergyHashList))
    if self.__active_hessian is not None and self.__active_hessian_ignore_hashes == ignored_hashes:
      return self.__active_hessian

    active_hessian: Optional[hessian] = None
    for request, current_hessian in zip(self.__energy_requests, self.__differentiated_hessians):
      if request.hash in self.__ignoredEnergyHashList:
        continue
      if active_hessian is None:
        active_hessian = current_hessian
      else:
        active_hessian = active_hessian + current_hessian

    for request, current_hessian in zip(self.__energy_requests_dynamic, self.__differentiated_hessians_dynamic):
      if request.hash in self.__ignoredEnergyHashList:
        continue
      if active_hessian is None:
        active_hessian = current_hessian
      else:
        active_hessian = active_hessian + current_hessian

    self.__active_hessian = active_hessian
    self.__active_hessian_ignore_hashes = ignored_hashes
    return self.__active_hessian

  @timed("minimizer.computeNumericValue")
  def computeNumericValue(self) -> Optional[hessian]:
    if len(self.__wrt) == 0:
      raise ValueError("minimizer.computeNumericValue: wrt is not initialized. Please call addWrt first.")
    if self.__gradient_object is None:
      self.__initializeGradientLayout()

    active_hessian = self.__getActiveHessian()
    if active_hessian is None:
      self.gradient.fill(0)
      self.__active_hessian = None
      return None

    active_hessian.compute(self.__gradient_object)
    self.__active_hessian = active_hessian
    self.__gradient_object = active_hessian.gradient

    if self.__diagonalBlockInverseKernel is None:
      self.__diagonalBlockInverseKernel = diagonalBlockInverseKernel(
        set([item.size for item in self.wrt]),
        active_hessian.diagonal_blocks_start_cpu,
        [item.correspondance.numInstances for item in self.__wrt],
        [item.size for item in self.__wrt],
        len(self.__wrt)
      )
    assert self.__diagonalBlockInverseKernel is not None
    if active_hessian.diagonal_blocks.size > 0:
      self.__diagonalBlockInverseKernel.computeDiagonalBlockInverse(active_hessian.diagonal_blocks, active_hessian.diagonal_blocks_inverse)
    return active_hessian

  def __ensureSolverBuffers(self) -> None:
    if self.gradient.size == 0:
      return
    if self.__solution.size == self.gradient.size and self.__solver is not None:
      return

    if self.__active_hessian is None:
      return

    self.__solver = solverKernel(self.__active_hessian.block_dimensions + self.__active_hessian.block_dimensions_dynamic)
    self.__d_p1_b = gpuarray.empty(self.gradient.shape, dtype=np.float64)
    self.__d_r = gpuarray.empty(self.gradient.shape, dtype=np.float64)
    self.__d_c = gpuarray.empty(self.gradient.shape, dtype=np.float64)
    self.__d_q = gpuarray.empty(self.gradient.shape, dtype=np.float64)
    self.__d_s = gpuarray.empty(self.gradient.shape, dtype=np.float64)
    self.__solution = gpuarray.empty(self.gradient.shape, dtype=np.float64)
    self.__last5solutions = gpuarray.empty(self.gradient.shape, dtype=np.float64)
    self.__solution.fill(0)
    self.__last5solutions.fill(0)
    self.__solutionSegments = []

    count = 0
    for segment in self.gradientSegments:
      self.__solutionSegments.append(self.__solution[count: count + segment.shape[0]])
      count += segment.shape[0]

  def __solveLinearSystem(self, tolerance = 1e-3, maxIterations = 20000):
    if self.gradient.size == 0:
      return 0
    if self.__active_hessian is None:
      self.__ensureSolverBuffers()
      if self.__solution.size > 0:
        self.__solution.fill(0)
      return 0

    self.__ensureSolverBuffers()
    assert self.__solver is not None
    self.__solver.updateBlockDimensions(self.__active_hessian.block_dimensions + self.__active_hessian.block_dimensions_dynamic)

    self.__d_p1_b.fill(0)
    self.__d_r.fill(0)
    self.__d_c.fill(0)
    self.__d_q.fill(0)
    self.__d_s.fill(0)
    self.__solution.fill(0)

    error_code = self.__solver.computeSolution(
      maxIterations,
      tolerance,
      self.__active_hessian.blocks_flattened,
      self.__active_hessian.block_positions,
      self.__active_hessian.blocks_start_indices,
      self.__active_hessian.block_counts,
      self.__active_hessian.block_dimensions,
      self.__active_hessian.blocks_flattened_dynamic,
      self.__active_hessian.block_positions_dynamic,
      self.__active_hessian.blocks_start_indices_dynamic,
      self.__active_hessian.block_counts_dynamic,
      self.__active_hessian.block_dimensions_dynamic,
      self.__active_hessian.diagonal,
      self.__active_hessian.diagonal_blocks_inverse,
      self.__active_hessian.diagonal_blocks_start_cpu,
      [item.correspondance.numInstances for item in self.__wrt],
      [item.size for item in self.__wrt],
      self.__gradient_object.gradient_segments_start_cpu,
      len(self.__wrt),
      self.gradient,
      self.__d_p1_b,
      self.__d_r,
      self.__d_c,
      self.__d_q,
      self.__d_s,
      self.__solution,
      self.__last5solutions
    )
    return error_code

  @timed("minimizer.computeSolution")
  def computeSolution(self, tolerance = 1e-3, maxIterations = 20000) -> List[gpuarray.GPUArray]:
    error_code = self.computeHessianAndGradient(tolerance=tolerance, maxIterations=maxIterations)
    if error_code < 0:
      print("Warning: solver did not converge. Returning the best solution found.")
    return self.solutionSegments

  def ignoreEnergies(self, energies: List[attribute]) -> None:
    self.__ignoredEnergyHashList = [e.hash for e in energies]
    self.__active_hessian = None
    self.__active_hessian_ignore_hashes = tuple()

  def computeHessianAndGradient(self, tolerance = 1e-3, maxIterations = 20000):
    self.computeNumericValue()
    return self.__solveLinearSystem(tolerance=tolerance, maxIterations=maxIterations)

  def computeTotalEnergy(self) -> float:
    total_energy = 0.0
    for request in self.__energy_requests:
      if request.hash in self.__ignoredEnergyHashList:
        continue
      total_energy += gpuarray.sum(request.energy.compute().value).get()
    for request in self.__energy_requests_dynamic:
      if request.energy.correspondance.numInstances > 0:
        if request.hash in self.__ignoredEnergyHashList:
          continue
        total_energy += gpuarray.sum(request.energy.compute().value).get()
    return total_energy
