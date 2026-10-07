# cython: language_level=3
"""Separate derivative producers sharing the assembly kernel's input arguments."""
import hashlib

from yasps.codeGenerator import codeGenerator


class hessianEvaluationRequest:
  def __init__(self, gradient, hessian=None, jacobian=None):
    self.gradient = gradient
    self.hessian = hessian
    self.jacobian = jacobian
    stages = [stage for stage in (gradient, hessian, jacobian) if stage is not None]
    for stage in stages:
      codeGenerator(stage).generateCode()
    kernels = {}
    for stage in stages:
      for kernel in stage.deviceKernel.dependents + [stage.deviceKernel]:
        kernels[kernel.kernelHeader] = kernel
    self.kernels = [kernels[key] for key in sorted(kernels)]
    for field in ("kernelDatas", "kernelConnectivity", "kernelPrimitiveUnions"):
      values = {item.fullName: item for stage in stages for item in getattr(stage.deviceKernel, field)}
      setattr(self, field, [values[key] for key in sorted(values)])
    self.hessianSize = 0 if hessian is None else hessian.size
    self.jacobianSize = 0 if jacobian is None else jacobian.size
    self.scratchSize = max(gradient.size, self.hessianSize + self.jacobianSize)
    signature = "staged_v1|" + "|".join(f"{stage.fullNameWithHash}:{stage.size}" for stage in stages)
    self.cacheKey = hashlib.sha256(signature.encode()).hexdigest()

  # return the function call for a chosen attribute
  # could be gradient, or hessian, or jacobian
  def call(self, stage, output):
    if stage is None:
      return ""
    kernel = stage.deviceKernel
    arguments = [item.code_generation_data_name for item in kernel.kernelDatas]
    arguments += [item.code_generation_index_name for item in kernel.kernelConnectivity]
    arguments += [item.code_generation_csr_name for item in kernel.kernelConnectivity if item.dimension == 0]
    arguments += [item.code_generation_counts_name for item in kernel.kernelPrimitiveUnions]
    return f"{stage.fullName}_device_function({', '.join(arguments + ['instance', output])});"

  def arguments(self):
    values = [item.value for item in self.kernelDatas]
    values += [item.value for item in self.kernelConnectivity]
    values += [item.compressedRows for item in self.kernelConnectivity if item.dimension == 0]
    values += [item.children_primitive_counts_gpu for item in self.kernelPrimitiveUnions]
    return values
