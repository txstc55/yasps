import yasps.attribute as ya
from yasps.connectivity import connectivity
from typing import Dict, List

class namedAttributeCodeGenerator:
  # generate the code for attribute
  def __init__(self, input: ya.attribute):
    self.__input: ya.attribute = input
    self.__order: List[ya.attribute] = [input]
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
        if current.name != "":
          # we will also generate the kernel for datas and named attribute
          # even though in reality we never call kernel for datas
          # the reason we generate this is to know what attributes are needed
          # for the kernel
          self.__order.append(current)
          # check if we have already generated the kernel for it
          if current.hash not in self.__childrenAttributeKernels:
            codeGenerator = namedAttributeCodeGenerator(current)
            codeGenerator.generateCode()
            self.__childrenAttributeKernels[current.hash] = current
        else:
          # doesnt have a name
          # probably just operations
          # we will go over all the children
          self.__order.append(current)
          self.__stack.extend(current.children)
      else:
        # when correspondance is different
        # there are couple of scenarios
        # 1. the correspondance is a scene or mesh, which means this is an operation done on scene or mesh attributes, we will allow that
        if current.correspondance.type == "scene" or current.correspondance.type == "mesh":
          if current.name != "":
            # data or named attributes
            self.__order.append(current)
            # check if we have already generated the kernel for it
            if current.hash not in self.__childrenAttributeKernels:
              codeGenerator = namedAttributeCodeGenerator(current)
              codeGenerator.generateCode()
              self.__childrenAttributeKernels[current.hash] = current
          else:
            # this is an operation on scene or mesh
            self.__order.append(current)
            self.__stack.extend(current.children)
        else:
          # this is an operation on other primitive attributes probably
          # this is only done through gather
          # so it must have a name at that point
          if current.name != "":
            # we dont add it to order
            # but we will create a code generator to generate the code for it
            if current.hash not in self.__childrenAttributeKernels:
              codeGenerator = namedAttributeCodeGenerator(current)
              codeGenerator.generateCode()
              self.__childrenAttributeKernels[current.hash] = current
          else:
            # this should not happen, raise an error
            raise ValueError("codeGenerator.__generateCodeOrder: actually going to a child without a name and not the same correspondance.")


  def generateCode(self) -> None:
    if self.__input.deviceKernel is not None:
      # nothing to do
      return

    # actually generate the code
    self.__generateCodeOrder()
    # now we do the code generation
    # reverse order since we want to generate the code from the bottom up
    code_strings: List[str] = []
    attribute_replacements: Dict[int, int] = {} # from hash to intermediate index
    num_intermediates = 0
    # go from bottom to top
    for current in self.__order[::-1]:
      # check if we have gone through this attribute
      # if not, fetch the data
      if current.hash not in attribute_replacements:
        if current.operator == ya.DATA:
          # we can safely retrieve the data for those attributes
          if current.size == 1:
            # safely retrieve the single data
            code_strings.append(f'''
  double {current.fullName}_local_data = {current.correspondance.fullName}_global_data[{current.correspondance.fullName}_index];
''')
          else:
            # retrieve the data and put it in a matrix locally
            # the map is not a copy operation so it's almost free
            code_strings.append(f'''
  double {current.fullName}_local_data_temp[{current.size}];
  {current.fullName}_device_function({current.fullName}_global_data, {current.fullName}_index, {current.fullName}_local_data_temp);
  Eigen::Map<Eigen::Matrix<double, {current.rows}, {current.cols}, Eigen::RowMajor>> {current.fullName}_local_data({current.fullName}_local_data_temp);
''')
          # mark this attribute as have been gone through
          attribute_replacements[current.hash] = -1
        elif current.operator == ya.GATHER:
          # need to generate the code for the gathering operation
          # we know the children must be a named attribute for the gathering operator
          children_attribute = current.children[0] # should only have one child
          code_strings.append(f'''
  double {current.fullName}_local_data_temp[{current.size}];
  for (unsigned int i = 0; i < {current.through.dimension}; i++){{
    # grab the index for the through attribute
    unsigned int {current.through.fullName}_index = {current.through.fullName}_global_indices[{current.through.fromPrimitive.fullName}_index * {current.through.dimension} + i];
    // now for each row, grab the data
    double {current.fullName}_local_data_row_temp[{int(current.size / children_attribute.size)}];
    {children_attribute.fullName}_device_function({", ".join([f'{x.fullName}_global_data' for x in children_attribute.deviceKernel.kernelDatas])}, {", ".join([f'{x.fullName}_global_indices' for x in children_attribute.deviceKernel.kernelConnectivity])}, {current.fullName}_local_data_row_temp);
    for (unsigned int j = 0; j < {int(current.size / children_attribute.size)}; j++){{ // copy the data
      {current.fullName}_local_data_temp[i * {int(current.size / children_attribute.size)} + j] = {current.fullName}_local_data_row_temp[j];
    }}
  }})
''')
          attribute_replacements[current.hash] = -1
        elif current.name != "":
          # the code is already generated
          # we need to call the kernel
          if current.size == 1:
            code_strings.append(f'''
  double {current.fullName}_local_data = 0.0;
  {current.fullName}_device_function({", ".join([f'{x.fullName}_global_data' for x in current.deviceKernel.kernelDatas])}, {", ".join([f'{x.fullName}_global_indices' for x in current.deviceKernel.kernelConnectivity])}, &{current.fullName}_local_data);
''')
          else:
            code_strings.append(f'''
  double {current.fullName}_local_data_temp[{current.size}];
  {current.fullName}_device_function({", ".join([f'{x.fullName}_global_data' for x in current.deviceKernel.kernelDatas])}, {", ".join([f'{x.fullName}_global_indices' for x in current.deviceKernel.kernelConnectivity])}, {current.fullName}_local_data_temp);
  // convert the data to a matrix
  Eigen::Map<Eigen::Matrix<double, {current.rows}, {current.cols}, Eigen::RowMajor>> {current.fullName}_local_data({current.fullName}_local_data_temp);
''')
          attribute_replacements[current.hash] = -1
        else:
          # this is a computation
          # we simply generate the code for it
          # for any operation, we know that the children must
          # have been gone through
          # so we can safely retrieve the data

          # first we get the data in either a matrix or a scalar
          attribute_name:str = ""
          if current.size == 1:
            attribute_name = f"double INTERMEDIATE_{num_intermediates}"
            num_intermediates += 1
          else:
            attribute_name = f"Eigen::Matrix<double, {current.rows}, {current.cols}, Eigen::RowMajor> INTERMEDIATE_{num_intermediates}"
            num_intermediates += 1
          if current.operator.type == 0:
            code_strings.append(f'''
  {attribute_name} = {current.operator.name}({self.getIntermediateName(current.children[0], attribute_replacements)});
''')
          elif current.operator.type == 1:
            if current
            code_strings.append(f'''
  {attribute_name} = {self.getIntermediateName(current.children[0], attribute_replacements)} {current.operator.name} {self.getIntermediateName(current.children[1], attribute_replacements)};
''')
          elif current.operator.type == 2:
            code_strings.append(f'''
  {attribute_name} = {current.operator.name}({", ".join([self.getIntermediateName(x, attribute_replacements) for x in current.children])});
''')
          else:
            if current.operator == ya.ARRAY_ACCESS:
              code_strings.append(f'''
  {attribute_name} = {self.getIntermediateName(current.children[0], attribute_replacements)}[{current.children[1].index_value}];
''')
            elif current.operator == ya.ARRAY:
              code_strings.append(f'''
  double INTERMEDIATE_{num_intermediates}_DATA[{current.size}] = {{{", ".join([self.getIntermediateName(x, attribute_replacements) for x in current.children])}}};
  Eigen::Map<Eigen::Matrix<double, {current.rows}, {current.cols}, Eigen::RowMajor>> INTERMEDIATE_{num_intermediates}(INTERMEDIATE_{num_intermediates}_DATA);
''')
            elif current.operator == ya.TRANSPOSE:
              code_strings.append(f'''
  {attribute_name} = {self.getIntermediateName(current.children[0], attribute_replacements)}.transpose();
''')
            elif current.operator == ya.BROADCAST_ADD:
              code_strings.append(f'''
  {attribute_name} = {self.getIntermediateName(current.children[0], attribute_replacements)}.array() + {self.getIntermediateName(current.children[1], attribute_replacements)};
''')
            elif current.operator == ya.ROW:
              code_strings.append(f'''
  {attribute_name} = {self.getIntermediateName(current.children[0], attribute_replacements)}.row({current.children[1].index_value});
''')
            elif current.operator == ya.COL:
              code_strings.append(f'''
  {attribute_name} = {self.getIntermediateName(current.children[0], attribute_replacements)}.col({current.children[1].index_value});
''')
            else:
              raise ValueError("codeGenerator.generateCode: unknown operator.")



  def getIntermediateName(self, attribute: ya.attribute, replacement: Dict[int, int]) -> str:
    if attribute.operator == ya.FLOAT:
      return str(attribute.float_value)
    # return the name of the intermediate value
    attribute_hash = attribute.hash
    if attribute_hash not in replacement:
      raise ValueError("codeGenerator.getIntermediateName: attribute hash not found in replacement.")
    if replacement[attribute_hash] == -1:
      return f"{attribute.fullName}_local_data"
    else:
      return f"INTERMEDIATE_{replacement[attribute_hash]}"

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
