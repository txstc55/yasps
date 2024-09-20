import yasps.attribute as ya
from yasps.connectivity import connectivity
from typing import Dict, List

class namedAttributeCodeGenerator:
  # generate the code for attribute
  def __init__(self, input: ya.attribute):
    self.__input: ya.attribute = input
    self.__order: List[ya.attribute] = []
    self.__stack: List[ya.attribute] = input.children
    self.__childrenAttributeKernels: Dict[int, ya.attribute] = {}

  def __generateCodeOrder(self) -> None:
    if self.__input.deviceKernel is not None:
      # nothing to do
      return
    # generate code for the attributes with names
    while self.__stack:
      current = self.__stack.pop()
      if current.operator == ya.FLOAT:
        pass
      elif current.operator == ya.INDEX:
        pass
      elif current.correspondance.fullName == self.__input.correspondance.fullName:
        if current.operator == ya.DATA:
          self.__order.append(current)
        elif current.name != "":
          # it is a named attribute
          # we dont need to go deeper
          # since we will generate the code for that child
          self.__order.append(current)
          codeGenerator = namedAttributeCodeGenerator(current)
          codeGenerator.generateCode()
          self.__childrenAttributeKernels[current.hash] = current
        else:
          self.__order.append(current)
          self.__stack.extend(current.children)
      else:
        if current.operator == ya.DATA:
          if current.correspondance.type == "scene" or current.correspondance.type == "mesh":
            # we can safely retrieve the data for those attributes
            self.__order.append(current)
          else:
            # retrieving data from other attribute is only done through gather
            pass
        elif current.name != "":
          codeGenerator = namedAttributeCodeGenerator(current)
          codeGenerator.generateCode()
          self.__childrenAttributeKernels[current.hash] = current
        else:
          # go over the children until we hit a named attribute
          self.__stack.extend(current.children)


  def generateCode(self) -> None:
    # actually generate the code
    self.__generateCodeOrder()
    # now we do the code generation
    # reverse order since we want to generate the code from the bottom up
    for attribute in self.__order[::-1]:
      if attribute.operator == ya.DATA:
        # we can safely retrieve the data for those attributes
        pass
      elif attribute.name != "":
        # we need to generate the code for the named attribute
        # we need to retrieve the data from the children
        # and then apply the kernel
        pass
      else:
        # we need to generate the code for the unnamed attribute
        # we need to retrieve the data from the children
        # and then apply the kernel
        pass


class codeGenerator:
  def __init__(self):
    self.__order: List[ya.attribute] = []
    self.__attributes_with_kernels: Dict[str, ya.attribute] = {} # generate and record all the attributes that has a kenrel to it

  @property
  def order(self)->List[ya.attribute]:
    return self.__order

  @property
  def attributesWithKernels(self)->Dict[str, ya.attribute]:
    return self.__attributes_with_kernels


  def generateCodeOrder(self, input: ya.attribute) -> None:
    # TODO: Implement expression simplification
    # TODO: Implement common subexpression elimination
    stack: List[ya.attribute] = [input]
    while stack:
      current: ya.attribute = stack.pop()
      if current.operator == ya.FLOAT:
        # there is no need to record constant values
        pass
      elif current.operator == ya.INDEX:
        pass
      elif current.correspondance.fullName == input.correspondance.fullName:
        # print("same correspondance", str(current))
        if current.operator == ya.DATA:
          self.__order.append(current)
          # self.__attributes_with_kernels[current.fullName] = current
        elif current.name != "":
          # stop at the first named attribute
          self.__order.append(current)
          self.__attributes_with_kernels[current.fullName] = current
          stack.extend(current.children)
        else:
          self.__order.append(current)
          stack.extend(current.children)
      else:
        if current.operator == ya.DATA:
          if current.correspondance.type == "scene" or current.correspondance.type == "mesh":
            # for scene or mesh data, it is fine to put it on the execution order
            # as we might need it later on
            self.__order.append(current)
          else:
            # for data that's an attribute
            # and the correspondance is not the same
            # this means there's a gathering most likely
            # we don't do anything about it
            pass
        elif current.name != "":
          # generate kernel for named attributes
          self.__attributes_with_kernels[current.fullName] = current
          stack.extend(current.children)
        else:
          stack.extend(current.children)

  def generateCodeForNamedAttribute(self, input: ya.attribute) -> None:
    if input.deviceKernel is not None:
      # we have generated the code before
      return

    seen_attributes: Dict[int, int] = {} # from a hash to the intermediate value index
    stack: List[ya.attribute] = [input]
    order: List[ya.attribute] = []
    while stack:
      current: ya.attribute = stack.pop()
      if current.operator == ya.FLOAT:
        # there is no need to record constant values
        pass
      elif current.operator == ya.INDEX:
        pass
      elif current.correspondance.fullName == input.correspondance.fullName:
        # print("same correspondance", str(current))
        if current.operator == ya.DATA:
          order.append(current)
        elif current.name != "":
          order.append(current)
          stack.extend(current.children)
        else:
          order.append(current)
          stack.extend(current.children)
      else:
        if current.operator == ya.DATA:
          if current.correspondance.type == "scene" or current.correspondance.type == "mesh":
            # for scene or mesh data, it is fine to put it on the execution order
            # as we might need it later on
            order.append(current)
          else:
            # for data that's an attribute
            # and the correspondance is not the same
            # this means there's a gathering most likely
            # we don't do anything about it
            pass
        elif current.name != "":
          order.append(current)
          stack.extend(current.children)
        else:
          stack.extend(current.children)


  def generateCodeForAttribute(self, input: ya.attribute) -> None:
    if input.deviceKernel is not None:
      return
    # first we generate the order
    # and extrate the attributes that needs to generate the kernels
    self.generateCodeOrder(input)
    # ok now we first go through the order of the named attributes

#   def generateCodeForAttribute(self, input: ya.attribute) -> None:
#     if input.deviceKernel != "":
#       return
#     else:
#       if input.operator == ya.DATA:
#         input_size: int = input.size # get the dimension of the data
#         input.deviceKernel = f'''
# __device__ double* compute_{input.fullName}(unsigned int index, double* {input.fullName}, double* output){{
#   for (unsigned int i = 0; i < {input_size}; i++){{
#     output[i] = {input.fullName}[i];
#   }}
#   return {input.fullName} + index * {input_size};
# }}
# '''


#   # recursively generate compute kernels for all children
#   # if they have a name
#   def generateCode(self, input: ya.attribute) -> None:
#     # store the kernels that have been generated
#     intermediate_kernels: Dict[int, str] = {}

#     # store the hashes if they have been seen
#     # from a hash of an attribute to the intermediate variable names
#     intermediate_replacement_hashes: Dict[int, str] = {}

#     # store the data attributes needed
#     # for kernel input lookup
#     data_needed: Dict[int, ya.attribute] = {}

#     # store the connectivity needed
#     # for kernel input lookup
#     connectivity_needed: Dict[str, connectivity] = {}

#     # store the strings for each node
#     code_strings: List[str] = []

#     # we first generate the code order
#     self.generateCodeOrder(input)

#     # traverse the order from last to first
#     for i in range(len(self.__order) - 1, -1, -1):
#       current: ya.attribute = self.__order[i]
#       current_hash: int = current.hash

#       if current.operator == ya.DATA:
#         if current_hash not in data_needed:
#           data_needed[current.hash] = current # record that we need this data
#           intermediate_replacement_hashes[current.hash] = f'INTERMEDIATE_{len(data_needed) - 1}' # record the intermediate number
#           # add the code for computing this data
#           code_strings.append(f'''
# Eigen::Map<Eigen::Matrix<double, {current.rows}, {current.cols}, Eigen::RowMajor>> {current.fullName}({current.fullName}_data + {current.fullName}_index * {current.rows} * {current.cols});
# ''')
#       elif current.operator == ya.GATHER:
#         # in a gather operation
#         # we need both datas
#         gathering_connectivity = current.through
#         if gathering_connectivity is None:
#           raise ValueError("codeGenerator.generateCode: Gather operation must have a through connectivity")
#           # check if the name is already in connectivity
#         if gathering_connectivity.fullName not in connectivity_needed:
#           connectivity_needed[gathering_connectivity.fullName] = gathering_connectivity
#         # check if this gathering kernel has been generated
#         if current.hash not in intermediate_kernels:
#           # we generate the kernel that specifically does the gathering
#           intermediate_kernels[current.hash] = f'''
# __device__ void gather_{gathering_connectivity.fullName}({', '.join([x.full_name + '_data' for x in data_needed.values()])}, {', '.join([x.fullName])})
# '''
#       else:
#         self.generateCodeForAttribute(current)
