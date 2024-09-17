import yasps.attribute as ya
from typing import Dict, List


class codeGenerator:
  def __init__(self):
    self.__order: List[ya.attribute] = []


  def generateCodeOrder(self, input: ya.attribute) -> None:
    # TODO: Implement expression simplification
    # TODO: Implement common subexpression elimination
    stack: List[ya.attribute] = [input]
    while stack:
      current: ya.attribute = stack.pop()
      self.__order.append(current)
      if current.operator == ya.DATA:
        self.__order.append(input)
      elif current.operator == ya.FLOAT:
        self.__order.append(input)
      elif current.name != "":
        # stop at the first named attribute
        self.__order.append(input)
      else:
        self.__order.extend(input.children)

  def generateCodeForAttribute(self, input: ya.attribute) -> None:
    if input.deviceKernel != "":
      return
    else:
      if input.operator == ya.DATA:
        input_size: int = input.size # get the dimension of the data
        input.deviceKernel = f'''
__device__ double* compute_{input.fullName}(unsigned int index, double* {input.fullName}, double* output){{
  for (unsigned int i = 0; i < {input_size}; i++){{
    output[i] = {input.fullName}[i];
  }}
  return {input.fullName} + index * {input_size};
}}
'''


  # recursively generate compute kernels for all children
  # if they have a name
  def generateCode(self, input: ya.attribute, kernels: set[str], headers: set[str], usedAttributes: set[ya.attribute]) -> None:
    # we first generate the code order
    self.generateCodeOrder(input)
    # now from bottom to top we generate the code
    for att in self.__order:
      pass
