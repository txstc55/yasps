# cython: language_level=3
from __future__ import annotations
from enum import IntFlag
import yasps.attribute as ya
from yasps.connectivity import connectivity
from yasps.primitiveUnion import primitiveUnion
from typing import Dict, List, Set, Tuple
from yasps.deviceKernel import deviceKernel
from yasps.globalKernel import globalKernel
from yasps.helper import timed

class codeGenerator:
  # generate the code for attribute
  def __init__(self, input: ya.attribute):
    self.__input: ya.attribute = input
    self.__order: List[ya.attribute] = []
    # print("------------------------------------------------------")
    # print("Checking input string")
    # print(str(input))
    # print("------------------------------------------------------")
    self.__stack: List[ya.attribute] = list(input.children)
    self.__childrenAttributeKernels: Dict[int, ya.attribute] = {}
    self.__attribute_replacements: Dict[int, Tuple[ya.attribute, int]] = {}
    self.__num_intermediates: int = 0
    self.__code_strings: List[str] = []
    self.__repeated_intermediates: Set[int] = set()
    self.__seen_elements: Set[int] = set()
    self.__seen_evd_projection_sizes: Set[int] = set() # used to track the projection size for evd
    self.__attribute_last_appear: Dict[int, int] = {} # when is this attribute last used
    self.__attribute_appear_times: Dict[int, int] = {} # store for each attribute, the number of times it appears
    self.__skipped_attribute_count: int = 0
    self.__current_order_index = -1 # this is basically the pointer to which node we are generating code currently
    self.__intermediate_dimensions: Dict[int, List[int]] = {} # store for each dimension, the list of intermediate attribute numbers
    self.__intermediate_last_appear_index: Dict[int, int] = {} # store for each intermediate number, its last appear index in the order
    self.__total_saved_registers = 0
    self.__total_initialized_registers = 0
    self.__seen_attribute_names: Dict[int, str] = {}
    self.__can_omit_intermediate_operators: List[ya.operator] = [] # for those operators, if they the symbolic only appears once, it is possible to omit them and directly put them inline [ya.NEG, ya.TRANSPOSE, ya.BROADCAST_ADD, ya.BROADCAST_SUB, ya.ROW, ya.COL, ya.CROSS, ya.NORM, ya.DET, ya.INV, ya.DOT]

  # storing when is the last time an attribute appear in the order array
  # this is done by checking the parent and children
  def __store_last_appear_time(self, att: ya.attribute) -> None:
    if att.hash not in self.__attribute_appear_times:
      self.__attribute_appear_times[att.hash] = 1
    else:
      self.__attribute_appear_times[att.hash] += 1
    if att.operator == ya.ARRAY_ACCESS:
      # in this case, we should store the last appear time for the actual array, instead of this specific element
      self.__attribute_last_appear[att.children[0].hash] = len(self.__order)
    else:
      if att.hash not in self.__attribute_last_appear:
        self.__attribute_last_appear[att.hash] = len(self.__order) # this is the time where all the children are free
      else:
        self.__attribute_last_appear[att.hash] = max(self.__attribute_last_appear[att.hash], len(self.__order)) # we take the max because we want to make sure we cover all the uses of this attribute

  def __generateCodeOrderDFS(self, current):
    # print("Generating code for attribute:", current.fullName)
    # print(current)
    if self.__input.deviceKernel is not None:
      # nothing to do
      return
    if current.hash in self.__seen_elements:
      return
    if current.operator == ya.FLOAT or current.operator == ya.INDEX:
      self.__attribute_last_appear[current.hash] = int(1e9)
      return
    elif current.isFloatMat:
      self.__attribute_last_appear[current.hash] = int(1e9)
      if current.correspondance is None:
        if current.operator == ya.TRANSPOSE:
          self.__generateCodeOrderDFS(current.children[0])
        self.__order.append(current)
      elif current.correspondance.fullName != self.__input.correspondance.fullName:
        # special case, we have a children from another correspondance
        # that is a float mat
        # this is likely because it's a unioned child or a joined child
        # so special handling is needed
        if current.hash not in self.__childrenAttributeKernels:
          childCodeGenerator = codeGenerator(current)
          childCodeGenerator.generateCode()
          self.__childrenAttributeKernels[current.hash] = current
      else:
        # special case for transpose
        if current.operator == ya.TRANSPOSE:
          self.__generateCodeOrderDFS(current.children[0])
        self.__order.append(current)
    elif current.operator == ya.SELECT:
      self.__generateCodeOrderDFS(current.children[0])
      self.__generateCodeOrderDFS(current.children[1])
      self.__generateCodeOrderDFS(current.children[2])
      self.__order.append(current)
      # mark last appearance
      for item in current.children:
        self.__store_last_appear_time(item)
    elif current.correspondance.fullName == self.__input.correspondance.fullName:
      if current.name != "" and current.generate_code:
        # this is a named attribute, lets use the generated kernel for it
        # instead of going over all its children
        self.__order.append(current)
        # check if we have already generated the kernel for it
        if current.hash not in self.__childrenAttributeKernels:
          childCodeGenerator = codeGenerator(current)
          childCodeGenerator.generateCode()
          self.__childrenAttributeKernels[current.hash] = current
        else:
          # there's a kernel generated that does exactly the same thing
          # we add a macro to handle this
          if current.fullName != self.__childrenAttributeKernels[current.hash].fullName:
            # we add a macro to make the two functions the same
            self.__code_strings.append(f'''
#define {current.fullName}_device_function {self.__childrenAttributeKernels[current.hash].fullName}_device_function
''')
      else:
        # we go over its children first
        for item in current.children:
          self.__generateCodeOrderDFS(item)
        # we went over the children, now we know if we need to put it in the order
        if current.hash not in self.__seen_elements:
          self.__seen_elements.add(current.hash)
          self.__order.append(current)
          for item in current.children:
            self.__store_last_appear_time(item)

    else:
      # when correspondance is different
      # there are couple of scenarios
      # 1. the correspondance is a scene or mesh, which means this is an operation done on scene or mesh attributes, we will allow that
      if current.correspondance.type == "mesh" or current.correspondance.type == "scene":
        if current.name != "":
          # data or named attributes
          self.__order.append(current)
          # check if we have already generated the kernel for it
          if current.hash not in self.__childrenAttributeKernels:
            childCodeGenerator = codeGenerator(current)
            childCodeGenerator.generateCode()
            self.__childrenAttributeKernels[current.hash] = current
        else:
          for item in current.children:
            self.__generateCodeOrderDFS(item)
          if current.hash not in self.__seen_elements:
            self.__seen_elements.add(current.hash)
            self.__order.append(current)
            for item in current.children:
              self.__store_last_appear_time(item)
      else:
        # this is an operation on other primitive attributes probably
        # this is done through join or union
        # so it must have a name at that point
        # if current.name != "":
        # we dont add it to order
        # but we will create a code generator to generate the code for it
        if current.hash not in self.__childrenAttributeKernels:
          childCodeGenerator = codeGenerator(current)
          childCodeGenerator.generateCode()
          self.__childrenAttributeKernels[current.hash] = current
    # print("Finished generating code for attribute:", current.fullName)


  def generateCode(self) -> None:
    if self.__input.deviceKernel is not None:
      # nothing to do
      return
    if len(self.__stack) == 0 and (self.__input.operator == ya.DATA or self.__input.operator == ya.CONSTANT):
      # this attribute is a data, generate special code for it
      current: ya.attribute = self.__input
      kernelString: str
      if current.correspondance.type == "scene" or current.correspondance.type == "mesh":
        kernelString: str = f'''
  #pragma unroll
  for (unsigned int i = 0; i < {current.size}; i++) {{
    result[i] = {current.code_generation_data_name}[i];
  }}'''
      else:
        kernelString: str = f'''
  #pragma unroll
  for (unsigned int i = 0; i < {current.size}; i++) {{
    result[i] = {current.code_generation_data_name}[{current.correspondance.fullName}_index * {current.size} + i];
  }}'''
      kernelHeader: str = f'''
__device__ void {current.fullName}_device_function(const double* {current.code_generation_data_name}, unsigned int {current.correspondance.fullName}_index, double* result)'''
      current.deviceKernel = deviceKernel(f'{kernelHeader}{{\n{kernelString}\n}}', kernelHeader, [current], [], [], [], set(), current.fullNameWithHash) # initialize the kernel with the code, the header, self as data, no connectivity, no dependents
      return
    for item in self.__input.children:
      self.__generateCodeOrderDFS(item)
    self.__order.append(self.__input)
    for item in self.__input.children:
      self.__store_last_appear_time(item)
    self.__attribute_last_appear[self.__input.hash] = len(self.__order) + 100 + len(self.__input.children)

    self.__current_order_index = -1 # this is basically the pointer to which node we are generating code currently
    # now generate the code
    if self.__input.size > 1:
      # pre-define this RowMat type
      self.__code_strings.append(f'''
#if {int(self.__input.rows == 1 or self.__input.cols == 1)}
  using RowMat = Eigen::Matrix<double,
                      {self.__input.rows},
                      {self.__input.cols}>;
#else
  using RowMat = Eigen::Matrix<double,
                    {self.__input.rows},
                    {self.__input.cols},
                    Eigen::RowMajor>;
#endif
  Eigen::Map<RowMat> out(result);
''')
    for current in self.__order:
      self.__current_order_index += 1
      if current.hash in self.__attribute_replacements:
        # we don't need to do anything about it
        continue

      if current.hash == self.__input.hash or current.name == "" or (current.name != "" and not current.generate_code):
        # we need to generate the code accordingly
        if current.name != "":
          # don't differentiate special cases, lets just use intermediate all the time
          self.__attribute_replacements[current.hash] = (current, self.__num_intermediates)
          self.__num_intermediates += 1
        else:
          self.__attribute_replacements[current.hash] = (current, self.__num_intermediates)
          self.__num_intermediates += 1

        # if current.hash in self.__attribute_appear_times and self.__attribute_appear_times[current.hash] == 1 and current != self.__input:
        #   if current.operator.type == -1 or current.operator.type == -1 or current.operator.type == -1 or current.operator in self.__can_omit_intermediate_operators:
        #     # if this operation only appears once, we can directly generate the code for it
        #     self.__skipped_attribute_count += 1
        #     self.__attribute_appear_times[current.hash] = -1
        #     # I know what the problem is
        #     # previously because we store all the intermediates, so the last appear time works
        #     # but now because we sometimes don't materialize the intermediates, so the last appear time should follow where this current "appear once" attribute is, before that attribute is deallocated, ok this is good
        #     continue

        # now we generate the computation code
        if current.operator.type == 0:
          self.__generate_code_for_type_0(current)
        elif current.operator.type == 1:
          self.__generate_code_for_type_1(current)
        elif current.operator.type == 2:
          self.__generate_code_for_type_2(current)
        else:
          # special operator with type 3
          if current.operator == ya.INDEX:
            self.__generate_code_for_index(current)
          if current.operator == ya.NEG:
            self.__generate_code_for_neg(current)
          elif current.operator == ya.FLOAT:
            self.__generate_code_for_float(current)
          elif current.operator == ya.ARRAY_ACCESS:
            self.__generate_code_for_array_access(current)
          elif current.operator == ya.DATA or current.operator == ya.CONSTANT:
            # the only reason we will reach here
            # is because we are at the root node
            # since data attribute should always have a name
            # but we already handled that case
            return
          elif current.operator == ya.ARRAY:
            self.__generate_code_for_array(current)
          elif current.operator == ya.JOIN:
            self.__generate_code_for_join(current)
          # elif current.operator == ya.SUM or current.operator == ya.AVERAGE:
          #   self.__generate_code_for_sum_and_average(current)
          elif current.operator == ya.TRANSPOSE:
            self.__generate_code_for_transpose(current)
          elif current.operator == ya.BROADCAST_ADD:
            self.__generate_code_for_broadcast_add(current)
          elif current.operator == ya.BROADCAST_SUB:
            self.__generate_code_for_broadcast_sub(current)
          elif current.operator == ya.ROW:
            self.__generate_code_for_row(current)
          elif current.operator == ya.COL:
            self.__generate_code_for_col(current)
          elif current.operator == ya.CROSS:
            self.__generate_code_for_cross(current)
          elif current.operator == ya.NORM:
            self.__generate_code_for_norm(current)
          elif current.operator == ya.DET:
            self.__generate_code_for_det(current)
          elif current.operator == ya.INV:
            self.__generate_code_for_inverse(current)
          elif current.operator == ya.DOT:
            self.__generate_code_for_dot(current)
          elif current.operator == ya.RESIZE:
            self.__generate_code_for_resize(current)
          elif current.operator == ya.SPD:
            self.__generate_code_for_spd(current)
          elif current.operator == ya.UNION:
            self.__generate_code_for_union(current)
      else:
        # it is not an output
        # and it has a name
        # so we can safely call the function
        self.__attribute_replacements[current.hash] = (current, self.__num_intermediates)
        self.__num_intermediates += 1
        self.__generate_code_for_named_attribute(current)

    # now we are done with the computation
    # need to put back the result
    attributeName: str = ""
    if self.__input.name == "":
      attributeName = self.__input.fullName
    else:
      attributeName = self.__input.fullName
#     # we have finished the computation, add a line to store the result
#     if self.__input.size == 1:
#       # for size 1 elements we can always compute then put it back
#       self.__code_strings.append(f'''
#   result[0] = {self.getIntermediateName(self.__input)};
# ''')
#     else:
#       self.__code_strings.append(f'''
#   out.noalias() = {self.getIntermediateName(self.__input)}.derived();   // Eigen decides whether a temp is needed
# ''')
    # now we need to get the datas for generating this kernel
    allNamedAttributeChildren = self.__childrenAttributeKernels.values()
    # get the datas they need
    allDatas: List[ya.attribute] = [item for x in allNamedAttributeChildren for item in x.deviceKernel.kernelDatas] # get all datas
    allConnectivities: List[connectivity] = [item for x in allNamedAttributeChildren for item in x.deviceKernel.kernelConnectivity] # get all the connectivities
    allPrimitiveUnions: List[primitiveUnion] = [item for x in allNamedAttributeChildren for item in x.deviceKernel.kernelPrimitiveUnions] # get all the primitive unions
    allDependencies: List[deviceKernel] = [item for x in allNamedAttributeChildren for item in x.deviceKernel.dependents] # get all the dependencies as strings
    allDependencies = allDependencies + [x.deviceKernel for x in allNamedAttributeChildren] # also add the children as dependencies

    # if we are a joining operation
    # we need to set the connectivity
    if self.__input.operator == ya.JOIN or self.__input.operator == ya.SUM or self.__input.operator == ya.AVERAGE:
      allConnectivities.append(self.__input.through)

    if self.__input.operator == ya.UNION:
      allPrimitiveUnions.append(self.__input.correspondance)

    # sort and remove duplicates
    allDatas = sorted(set(allDatas), key = lambda x: x.fullName)
    allConnectivities = sorted(set(allConnectivities), key = lambda x: x.fullName)
    allDependencies = sorted(set(allDependencies), key = lambda x: x.kernelHeader)
    allPrimitiveUnions = sorted(set(allPrimitiveUnions), key = lambda x: x.fullName)
    allEvdSizes = self.__seen_evd_projection_sizes
    for dependency in allDependencies:
      allEvdSizes.update(dependency.allEvdSizes)


    # now we generate header
    headerString: str = f'''
__device__ void {attributeName}_device_function(
  {"".join([f"const double* {x.code_generation_data_name}, " for x in allDatas])}
  {"".join([f"const unsigned int* {x.code_generation_index_name}, " for x in allConnectivities])}
  {"".join([f"const unsigned int* {x.code_generation_csr_name}, " for x in allConnectivities if x.dimension == 0])}
  {"".join([f'const unsigned int* {x.code_generation_counts_name},' for x in allPrimitiveUnions])}
  const unsigned int {self.__input.correspondance.fullName}_index,
  double* result
)'''

    # remove the duplicates, some of the index initialization may be duplicated
    # seen_strings: Set[str] = set()
    self.__code_strings.append(f'''
// We initialized {self.__total_initialized_registers} registers
// We saved {self.__total_saved_registers} registers by reusing intermediate variables
// There are {self.__skipped_attribute_count} skipped attributes
'''
)
    self.__code_strings = [x.strip('\n') for x in self.__code_strings]

    kernelString: str = "\n".join(self.__code_strings)

    # now we generate the device kernel
    self.__input.deviceKernel = deviceKernel(f'{headerString}{{\n{kernelString}\n}}', headerString, allDatas, allConnectivities, allPrimitiveUnions, allDependencies, allEvdSizes, self.__input.fullNameWithHash)
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # print(f"We initialized {self.__total_initialized_registers} registers")
    # print(f"We saved {self.__total_saved_registers} registers")
    # print(f"There are {self.__skipped_attribute_count} skipped attributes")
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

  # get the name of the intermediate variables
  def getIntermediateName(self, attribute: ya.attribute) -> str:
    if attribute.hash in self.__attribute_appear_times and self.__attribute_appear_times[attribute.hash] == -1:
      if attribute.operator.type == 0:
        return self.__generate_code_for_type_0(attribute, True)
      elif attribute.operator.type == 1:
        return self.__generate_code_for_type_1(attribute, True)
      elif attribute.operator.type == 2:
        return self.__generate_code_for_type_2(attribute, True)
      else:
        if attribute.operator == ya.NEG:
          return self.__generate_code_for_neg(attribute, True)
        elif attribute.operator == ya.TRANSPOSE:
          return self.__generate_code_for_transpose(attribute, True)
        elif attribute.operator == ya.BROADCAST_ADD:
          return self.__generate_code_for_broadcast_add(attribute, True)
        elif attribute.operator == ya.BROADCAST_SUB:
          return self.__generate_code_for_broadcast_sub(attribute, True)
        elif attribute.operator == ya.ROW:
          return self.__generate_code_for_row(attribute, True)
        elif attribute.operator == ya.COL:
          return self.__generate_code_for_col(attribute, True)
        elif attribute.operator == ya.CROSS:
          return self.__generate_code_for_cross(attribute, True)
        elif attribute.operator == ya.NORM:
          return self.__generate_code_for_norm(attribute, True)
        elif attribute.operator == ya.DET:
          return self.__generate_code_for_det(attribute, True)
        elif attribute.operator == ya.INV:
          return self.__generate_code_for_inverse(attribute, True)
        elif attribute.operator == ya.DOT:
          return self.__generate_code_for_dot(attribute, True)

    if attribute.hash in self.__seen_attribute_names:
      return self.__seen_attribute_names[attribute.hash]
    if attribute.operator == ya.FLOAT:
      self.__seen_attribute_names[attribute.hash] = str(attribute.float_value)
      return str(attribute.float_value)
    # return the name of the intermediate value

    attribute_hash = attribute.hash
    if attribute.operator == ya.SPD and attribute.children[0].size > 1:
      # for spd, let's do inplace projection
      # but for size 1, we don't need to do it in place, instead we can just allocate a new intermediate variable
      attribute_hash = attribute.children[0].hash
    if attribute_hash not in self.__attribute_replacements:
      raise ValueError(f"codeGenerator.getIntermediateName: attribute hash not found in self.__attribute_replacements. {str(attribute)} hash is: {attribute_hash}")
    if attribute.operator == ya.ARRAY_ACCESS:
      # for array access, we return the name of the actual array
      index_value = attribute.children[1].index_value
      row_num = index_value // attribute.children[0].cols
      col_num = index_value % attribute.children[0].cols
      child_rows = attribute.children[0].rows
      child_cols = attribute.children[0].cols
      mat_name = self.getIntermediateName(attribute.children[0])
      if child_rows == 1 and child_cols == 1:
        # this is a direct access to a scalar
        self.__seen_attribute_names[attribute.hash] = mat_name
        return mat_name
      else:
        self.__seen_attribute_names[attribute.hash] = f'{mat_name}({row_num}, {col_num})'
        return f'{mat_name}({row_num}, {col_num})'
    if self.__attribute_replacements[attribute_hash][1] == -1:
      self.__seen_attribute_names[attribute.hash] = f"{self.__attribute_replacements[attribute_hash][0].fullName}_local_data"
      return f"{self.__attribute_replacements[attribute_hash][0].fullName}_local_data"
    else:
      self.__seen_attribute_names[attribute.hash] = f"INTERMEDIATE_{self.__attribute_replacements[attribute_hash][1]}"
      return f"INTERMEDIATE_{self.__attribute_replacements[attribute_hash][1]}"






  ################################################
  ################################################
  ################################################
  ################################################
  #     CODE GENERATION FOR DIFFERENT OPERATORS
  ################################################
  ################################################
  ################################################
  ################################################

  def __check_available_intermediate_number(self, att: ya.attribute):
    dimension = att.rows * att.cols
    if dimension not in self.__intermediate_dimensions:
      self.__intermediate_dimensions[dimension] = [self.__attribute_replacements[att.hash][1]]
      self.__intermediate_last_appear_index[self.__attribute_replacements[att.hash][1]] = self.__attribute_last_appear[att.hash]
      self.__total_initialized_registers += att.size
      return -self.__attribute_replacements[att.hash][1] - 1
    else:
      if att.operator != ya.RESIZE:
        for ind in self.__intermediate_dimensions[dimension]:
          if self.__intermediate_last_appear_index[ind] <= self.__current_order_index:
            # we can reuse this intermediate variable
            if dimension == 1:
              self.__attribute_replacements[att.hash] = (att, ind) # we will directly use that singular register
              # note that we don't do this when dimension is not equal to 1
              # we only need to reuse the storage space
              # but we can still give it a new name
            # if att.hash not in self.__attribute_last_appear:
            #   print(str(att))
            self.__intermediate_last_appear_index[ind] = self.__attribute_last_appear[att.hash]
            self.__total_saved_registers += att.size
            return ind
      # if we cannot find any available intermediate variable, we need to create a new one
      self.__intermediate_dimensions[dimension].append(self.__attribute_replacements[att.hash][1])
      if att.hash not in self.__attribute_last_appear:
        print(str(att))
      self.__intermediate_last_appear_index[self.__attribute_replacements[att.hash][1]] = self.__attribute_last_appear[att.hash]
      self.__total_initialized_registers += att.size
      return -self.__attribute_replacements[att.hash][1] - 1

  def __generate_attribute_name_and_initialization(self, current: ya.attribute) -> Tuple[str, str]:
    attribute_initialization: str = ""
    attribute_name: str = ""
    # if current.name != "":
    #   attribute_name = current.fullName + "_local_data"
    #   self.__attribute_replacements[current.hash] = (current, -1)
    #   if current.size == 1:
    #     attribute_initialization = f"double {attribute_name}"
    #   else:
    #     if current.rows == 1 or current.cols == 1:
    #       attribute_initialization = f"Eigen::Matrix<double, {current.rows}, {current.cols}> {attribute_name}"
    #     else:
    #       attribute_initialization = f"Eigen::Matrix<double, {current.rows}, {current.cols}, Eigen::RowMajor> {attribute_name}"
    #   return attribute_name, attribute_initialization

    # ok we will do two cases
    # the first one is the current size is just 1
    if current.size == 1:
      # what we will do is we check if there's any available intermediate variables that we can reuse, if not we will just create the new one
      # otherwise we reuse
      free_register_result = self.__check_available_intermediate_number(current)
      if free_register_result < 0:
        # if the result is negative, it means we need to create a new intermediate variable
        ind: int = self.__attribute_replacements[current.hash][1]
        attribute_name = f"INTERMEDIATE_{ind}"
        attribute_initialization = f"double {attribute_name}"
      else:
        # if the result is non-negative, it means we can reuse an existing intermediate variable
        attribute_name = f"INTERMEDIATE_{free_register_result}"
        attribute_initialization = attribute_name
      return attribute_name, attribute_initialization
    else:
      # this is the case where the current size is larger than 1
      # we first check if there's any available intermediate variables that we can reuse for storage
      free_register_result = self.__check_available_intermediate_number(current)
      if free_register_result < 0:
        ind: int = self.__attribute_replacements[current.hash][1]
        attribute_name = f"INTERMEDIATE_{ind}"
        if current.rows == 1 or current.cols == 1:
          attribute_initialization = f"Eigen::Matrix<double, {current.rows}, {current.cols}> {attribute_name}"
        else:
          attribute_initialization = f"Eigen::Matrix<double, {current.rows}, {current.cols}, Eigen::RowMajor> {attribute_name}"
      else:
        # here's the part where we need to reuse an existing intermediate matrix's storage
        ind: int = self.__attribute_replacements[current.hash][1]
        attribute_name = f"INTERMEDIATE_{ind}"
        matrix_initialization = f"Eigen::Matrix<double, {current.rows}, {current.cols}>" if (current.rows == 1 or current.cols == 1) else f"Eigen::Matrix<double, {current.rows}, {current.cols}, Eigen::RowMajor>"
        # we initialize the matrix here by reusing the storage of the existing intermediate matrix
        self.__code_strings.append(f'''
  Eigen::Map<{matrix_initialization}> {attribute_name}(INTERMEDIATE_{free_register_result}.data());
''')
        attribute_initialization = attribute_name
      return attribute_name, attribute_initialization


  def __generate_code_for_type_0(self, current: ya.attribute, compute_only: bool = False) -> str:
    if compute_only:
      if current.size == 1:
        return f'{current.operator.name}({self.getIntermediateName(current.children[0])})'
      return f'({self.getIntermediateName(current.children[0])}).array().{current.operator.name}()'
    attribute_name, attribute_initialization = self.__generate_attribute_name_and_initialization(current)
    if current == self.__input:
      attribute_initialization = "result[0]" if current.size == 1 else "out.noalias()"
    if current.size == 1:
      self.__code_strings.append(f'''
  {attribute_initialization} = {current.operator.name}({self.getIntermediateName(current.children[0])});''')
      return ""
    else:
      self.__code_strings.append(f'''
  // allocate the space since this operation is most likely going to be expansive
  {attribute_initialization} = ({self.getIntermediateName(current.children[0])}).array().{current.operator.name}();''')
      return ""



  def __generate_code_for_type_1(self, current: ya.attribute, compute_only: bool = False) -> str:
    if compute_only:
      return f'(({self.getIntermediateName(current.children[0])}) {current.operator.name} ({self.getIntermediateName(current.children[1])}))'
    attribute_name, attribute_initialization = self.__generate_attribute_name_and_initialization(current)
    if current == self.__input:
      attribute_initialization = "result[0]" if current.size == 1 else "out.noalias()"

    self.__code_strings.append(f'''
  {attribute_initialization} = ({self.getIntermediateName(current.children[0])}) {current.operator.name} ({self.getIntermediateName(current.children[1])});''')
    return ""


  def __generate_code_for_type_2(self, current: ya.attribute, compute_only: bool = False) -> str:
    if compute_only:
      if current.operator == ya.SELECT:
        return f'(({self.getIntermediateName(current.children[0])}) ? ({self.getIntermediateName(current.children[1])}) : ({self.getIntermediateName(current.children[2])}))'
      else:
        return f'({current.operator.name}({", ".join([self.getIntermediateName(x) for x in current.children])}))'

    attribute_name, attribute_initialization = self.__generate_attribute_name_and_initialization(current)
    # for type 2 there isn't many operators
    # currently we have select or power
    # and power is forbidden on matrix
    # so we can always do it this way as op(a, b, c, d, ...)
    if current == self.__input:
      attribute_initialization = "result[0]" if current.size == 1 else "out.noalias()"
    if current.operator == ya.SELECT:
      self.__code_strings.append(f'''
  {attribute_initialization} = ({self.getIntermediateName(current.children[0])}) ? ({self.getIntermediateName(current.children[1])}) : ({self.getIntermediateName(current.children[2])});''')
      return ""
    else:
      self.__code_strings.append(f'''
  {attribute_initialization} = {current.operator.name}({", ".join(['(' + self.getIntermediateName(x) + ')' for x in current.children])});''')
      return ""


  def __generate_code_for_index(self, current: ya.attribute) -> str:
    # this should never ever happen
    raise ValueError("codeGenerator.generateCode: INDEX operator should never be reached.")

  def __generate_code_for_neg(self, current: ya.attribute, compute_only: bool = False) -> str:
    if compute_only:
      return f'(-({self.getIntermediateName(current.children[0])}))'
    attribute_name, attribute_initialization = self.__generate_attribute_name_and_initialization(current)
    if current == self.__input:
      attribute_initialization = "result[0]" if current.size == 1 else "out.noalias()"

    self.__code_strings.append(f'''
  {attribute_initialization} = -({self.getIntermediateName(current.children[0])});''')
    return ""

  def __generate_code_for_float(self, current: ya.attribute) -> str:
    # the float attribute can only happen if it is a root node
    # because when traversing we never put it on the stack
    self.__code_strings.append(f'''
  result[0] = {current.float_value};''')
    return ""

  def __generate_code_for_array_access(self, current: ya.attribute) -> str:
  # for array access we do nothing unless it is an output
    if current == self.__input:
      attribute_initialization = "result[0]"
      self.__code_strings.append(f'''
  {attribute_initialization} = {self.getIntermediateName(current)};''')
    return ""


  def __generate_code_for_array(self, current: ya.attribute) -> str:
    attribute_name, attribute_initialization = self.__generate_attribute_name_and_initialization(current)
    if current == self.__input:
      attribute_initialization = "out"
      attribute_name = attribute_initialization
    # generate code for array
    self.__code_strings.append(f'''
  {attribute_initialization};
  // appending value for array operation
  {attribute_name} << {", ".join(['(' + self.getIntermediateName(x) + ')' for x in current.children])};
''')
    return ""







  def __generate_code_for_transpose(self, current: ya.attribute, compute_only: bool = False) -> str:
    if compute_only:
      return f'({self.getIntermediateName(current.children[0])}).transpose()'
    attribute_name, attribute_initialization = self.__generate_attribute_name_and_initialization(current)
    if current == self.__input:
      attribute_initialization = "result[0]" if current.size == 1 else "out"


    self.__code_strings.append(f'''
  {attribute_initialization} = ({self.getIntermediateName(current.children[0])}).transpose();''')
    return ""

  def __generate_code_for_broadcast_add(self, current: ya.attribute, compute_only: bool = False) -> str:
    if compute_only:
      return f'({self.getIntermediateName(current.children[0])}).array() + ({self.getIntermediateName(current.children[1])}).array()'
    attribute_name, attribute_initialization = self.__generate_attribute_name_and_initialization(current)
    if current == self.__input:
      attribute_initialization = "result[0]" if current.size == 1 else "out"

    self.__code_strings.append(f'''
  // broadcast addition
  {attribute_initialization} = ({self.getIntermediateName(current.children[0])}).array() + ({self.getIntermediateName(current.children[1])});''')
    return ""

  def __generate_code_for_broadcast_sub(self, current: ya.attribute, compute_only: bool = False) -> str:
    if compute_only:
      return f'({self.getIntermediateName(current.children[0])}).array() - ({self.getIntermediateName(current.children[1])}).array()'

    attribute_name, attribute_initialization = self.__generate_attribute_name_and_initialization(current)
    if current == self.__input:
      attribute_initialization = "result[0]" if current.size == 1 else "out"

    self.__code_strings.append(f'''
  // broadcast subtraction
  {attribute_initialization} = ({self.getIntermediateName(current.children[0])}).array() - ({self.getIntermediateName(current.children[1])});''')
    return ""

  def __generate_code_for_row(self, current: ya.attribute, compute_only: bool = False) -> str:
    if compute_only:
      return f'({self.getIntermediateName(current.children[0])}).row({current.children[1].index_value})'

    attribute_name, attribute_initialization = self.__generate_attribute_name_and_initialization(current)
    if current == self.__input:
      attribute_initialization = "result[0]" if current.size == 1 else "out"

    if current.children[0].cols == 1:
      # this is a singular value
      self.__code_strings.append(f'''
  // getting row to a singular value
  {attribute_initialization} = ({self.getIntermediateName(current.children[0])}).row({current.children[1].index_value}).value();''')
    else:
      self.__code_strings.append(f'''
  // getting row by expression template
  {attribute_initialization} = ({self.getIntermediateName(current.children[0])}).row({current.children[1].index_value});''')
    return ""

  def __generate_code_for_col(self, current: ya.attribute, compute_only: bool = False) -> str:
    if compute_only:
      return f'({self.getIntermediateName(current.children[0])}).col({current.children[1].index_value})'

    attribute_name, attribute_initialization = self.__generate_attribute_name_and_initialization(current)
    if current == self.__input:
      attribute_initialization = "result[0]" if current.size == 1 else "out"

    if current.children[0].rows == 1:
      # this is a singular value
      self.__code_strings.append(f'''
  // getting column to a singular value
  {attribute_initialization} = ({self.getIntermediateName(current.children[0])}).col({current.children[1].index_value}).value();''')
    else:
      # this is a column vector
      # we can use the expression template
      self.__code_strings.append(f'''
  // getting column by expression template
  {attribute_initialization} = ({self.getIntermediateName(current.children[0])}).col({current.children[1].index_value});''')
    return ""

  def __generate_code_for_cross(self, current: ya.attribute, compute_only: bool = False) -> str:
    if compute_only:
      return f'({self.getIntermediateName(current.children[0])}).cross(({self.getIntermediateName(current.children[1])}))'

    attribute_name, attribute_initialization = self.__generate_attribute_name_and_initialization(current)
    if current == self.__input:
      attribute_initialization = "result[0]" if current.size == 1 else "out"

    self.__code_strings.append(f'''
  {attribute_initialization} = ({self.getIntermediateName(current.children[0])}).cross(({self.getIntermediateName(current.children[1])}));''')
    return ""

  def __generate_code_for_norm(self, current: ya.attribute, compute_only: bool = False) -> str:
    if compute_only:
      return f'({self.getIntermediateName(current.children[0])}).norm()'

    attribute_name, attribute_initialization = self.__generate_attribute_name_and_initialization(current)
    if current == self.__input:
      attribute_initialization = "result[0]" if current.size == 1 else "out"

    self.__code_strings.append(f'''
  {attribute_initialization} = ({self.getIntermediateName(current.children[0])}).norm();''')
    return ""

  def __generate_code_for_det(self, current: ya.attribute, compute_only: bool = False) -> str:
    if compute_only:
      return f'({self.getIntermediateName(current.children[0])}).determinant()'

    attribute_name, attribute_initialization = self.__generate_attribute_name_and_initialization(current)
    if current == self.__input:
      attribute_initialization = "result[0]" if current.size == 1 else "out"

    self.__code_strings.append(f'''
  {attribute_initialization} = ({self.getIntermediateName(current.children[0])}).determinant();''')
    return ""

  def __generate_code_for_inverse(self, current: ya.attribute, compute_only: bool = False) -> str:
    if compute_only:
      return f'({self.getIntermediateName(current.children[0])}).inverse()'

    attribute_name, attribute_initialization = self.__generate_attribute_name_and_initialization(current)
    if current == self.__input:
      attribute_initialization = "result[0]" if current.size == 1 else "out"

    self.__code_strings.append(f'''
  {attribute_initialization} = ({self.getIntermediateName(current.children[0])}).inverse();''')
    return ""

  def __generate_code_for_dot(self, current: ya.attribute, compute_only: bool = False) -> str:
    if compute_only:
      return f'({self.getIntermediateName(current.children[0])}).dot(({self.getIntermediateName(current.children[1])}))'

    attribute_name, attribute_initialization = self.__generate_attribute_name_and_initialization(current)
    if current == self.__input:
      attribute_initialization = "result[0]" if current.size == 1 else "out"

    self.__code_strings.append(f'''
  {attribute_initialization} = ({self.getIntermediateName(current.children[0])}).dot(({self.getIntermediateName(current.children[1])}));''')
    return ""

  def __generate_code_for_resize(self, current: ya.attribute) -> str:
    # if this is an output, then we just need to put the data back
    if current == self.__input:
      # attribute_initialization_original = "result[0]" if current.size == 1 else "out"
      if current.size == 1:
        self.__code_strings.append(f'''
  result[0] = {self.getIntermediateName(current.children[0])};''')
      else:
        self.__code_strings.append(f'''
  out.noalias() = {self.getIntermediateName(current.children[0])};''')

    # # we need to generate the code accordingly
    # origin_mat = current.children[0]
    attribute_name, attribute_initialization_original = self.__generate_attribute_name_and_initialization(current)
    # if origin_mat.rows == 1 or origin_mat.cols == 1:
    #   attribute_initialization = f"Eigen::Matrix<double, {origin_mat.rows}, {origin_mat.cols}> {attribute_name}_before_resize = {self.getIntermediateName(current.children[0])};\n  "
    # else:
    #   attribute_initialization = f"Eigen::Matrix<double, {origin_mat.rows}, {origin_mat.cols}, Eigen::RowMajor> {attribute_name}_before_resize = {self.getIntermediateName(current.children[0])};\n  "
    # attribute_initialization += f'''{attribute_initialization_original}'''
    # let's assume that after resize it always replace the previous one
    # or at least they share the data regardless
    self.__code_strings.append(f'''
  // resize operation
  {attribute_initialization_original}(({self.getIntermediateName(current.children[0])}).data());''')
    return ""

  # MARK ISSUE
  # When the projection is 0, there is absolutely no need to allocate a matrix
  # but there's not a good workaround for this right now, since the compiler requires static memory allocation
  # We can always fall back to projection inplace, but that still requires some allocation
  def __generate_code_for_spd(self, current: ya.attribute) -> str:
    attribute_name, attribute_initialization = self.__generate_attribute_name_and_initialization(current)
    an = self.getIntermediateName(current.children[0]) # attribute being projected
    mn = self.getIntermediateName(current.children[1]) # attribute being used for projection method
    if current.size == 1:
      self.__code_strings.append(f'''
         {attribute_initialization} = {mn} == 0 ? {an} : ({mn} == 1 ? abs({an}) : max(0.0, {an}));''')
    else:
      if current.rows <= 3:
        self.__seen_evd_projection_sizes.add(current.rows)
        self.__code_strings.append(f'''
  // {attribute_initialization} = {mn} == 0 ? Eigen::Matrix<double, {current.rows}, {current.cols}, Eigen::RowMajor>({an}.data()) : Eigen::Matrix<double, {current.rows}, {current.cols}, Eigen::RowMajor>::Zero();
  if ({mn} != 0) {{
    spd_projection_small<{current.children[0].rows}>({an}.data(), {an}.data(), int({mn}));
  }}
  ''')
      else:
        # we do the normal projection
        self.__seen_evd_projection_sizes.add(current.rows)
        self.__code_strings.append(f'''
  // {attribute_initialization} = {mn} == 0 ? Eigen::Matrix<double, {current.rows}, {current.cols}, Eigen::RowMajor>({an}.data()) : Eigen::Matrix<double, {current.rows}, {current.cols}, Eigen::RowMajor>::Zero();
  if ({mn} != 0) {{
    spd_projection_inplace<{current.children[0].rows}>({an}.data(), int({mn}));
  }}
''')
    if current == self.__input:
      if current.size == 1:
        self.__code_strings.append(f'''
  result[0] = {attribute_initialization};''')
      else:
        self.__code_strings.append(f'''
  out.noalias() = {an};''')
    return ""


  def __generate_code_for_join(self, current: ya.attribute) -> str:
    attribute_name, attribute_initialization = self.__generate_attribute_name_and_initialization(current)
    if current == self.__input:
      attribute_initialization = "result[0]" if current.size == 1 else "out"
      attribute_name = attribute_initialization
    # need to generate the code for the joining operation
    # we know the children must be a named attribute for the joining operator
    children_attribute = current.children[0] # should only have one child
    children_attribute_name: str = ""
    if children_attribute.deviceKernel is None:
      # this is actually a very special case
      # we are joining a number
      # this can happen inside the hessian
      float_values = []
      for i in range(children_attribute.size):
        float_values.append(str(children_attribute[i].float_value))
      float_values = float_values * current.through.dimension
      float_values_string = ", ".join(float_values)
      if current.size == 1:
        self.__code_strings.append(f'''
  {attribute_initialization} = {float_values_string};
''')
      else:
        self.__code_strings.append(f'''
  {attribute_initialization};
  // appending value for JOIN operation with float values
  {attribute_name} << {float_values_string};
''')
      return
    if children_attribute.name == "":
      children_attribute_name = children_attribute.fullName
    else:
      children_attribute_name = children_attribute.fullName
    self.__code_strings.append(f'''
  {attribute_initialization};
  for (unsigned int i = 0; i < {current.through.dimension}; i++){{
  // grab the index for the through attribute
    unsigned int {current.through.fullName}_index = {current.through.code_generation_index_name}[{current.through.fromPrimitive.fullName}_index * {current.through.dimension} + i];
    // now for each row, grab the data
    double {current.fullName}_local_data_row_temp[{children_attribute.size}] = {{0}};
    {children_attribute_name}_device_function(
      {"".join([f'{x.code_generation_data_name}, ' for x in sorted(children_attribute.deviceKernel.kernelDatas, key = lambda y: y.fullName)])}
      {"".join([f'{x.code_generation_index_name}, ' for x in sorted(children_attribute.deviceKernel.kernelConnectivity, key = lambda y: y.fullName)])}
      {"".join([f'{x.code_generation_csr_name}, ' for x in children_attribute.deviceKernel.kernelConnectivity if x.dimension == 0])}
      {"".join([f'{x.code_generation_counts_name},' for x in sorted(children_attribute.deviceKernel.kernelPrimitiveUnions, key = lambda y: y.fullName)])}
      {current.through.fullName}_index,
      {f'&{attribute_name}' if current.size == 1 else attribute_name + f".data() + i * {int(children_attribute.size)}"}
    );
  }}
''')
    return ""


  def __generate_code_for_union(self, current: ya.attribute) -> str:
    attribute_name, attribute_initialization = self.__generate_attribute_name_and_initialization(current)
    if current == self.__input:
      attribute_initialization = "result[0]" if current.size == 1 else "out"
      attribute_name = attribute_initialization
    # for union operator, we just need to pick the correct path
    code_string = ""
    code_string += f'''
  {attribute_initialization};
  // we need to determine which primitive it's calling from
  {{
    unsigned int {current.fullName}_primitive_index = 0;
    unsigned int {current.fullName}_primitive_total_counts[{len(current.children)} + 1] = {{0}}; // to help us determine which child primitive to invoke
    // do a prefix sum
    for (unsigned int i = 0; i < {len(current.children)}; i++){{
      {current.fullName}_primitive_total_counts[i + 1] = {current.fullName}_primitive_total_counts[i] + {current.correspondance.code_generation_counts_name}[i];
    }}
    for (unsigned int i = 0; i < {len(current.children)}; i++){{
      if ({current.correspondance.fullName}_index >= {current.fullName}_primitive_total_counts[i] && {current.correspondance.fullName}_index < {current.fullName}_primitive_total_counts[i + 1]){{
        {current.fullName}_primitive_index = i;
        break;
      }}
    }}
    // now that we know the exact primitive index, we invoke the attribute function
    switch({current.fullName}_primitive_index){{
'''
    for i in range(len(current.children)):
      child_attribute = current.children[i]
      code_string += f'''
      case {i}:
'''
      if child_attribute.isFloatMat or child_attribute.operator == ya.FLOAT:
        # this is likely a constant value
        # we can just get the constant value I guess
        if child_attribute.size == 1:
          code_string += f'''
        {attribute_name} = {str(child_attribute.float_value)};
        break;
'''
        else:
          float_mat_string = ", ".join([str(child_attribute[j].float_value) for j in range(child_attribute.size)])
          code_string += f'''
        // appending value for UNION float mat
        {attribute_name} << {float_mat_string};
        break;
'''
      else:
        code_string += f'''
        {current.children[i].fullName}_device_function(
          {"".join([f'{x.code_generation_data_name}, ' for x in sorted(child_attribute.deviceKernel.kernelDatas, key = lambda y: y.fullName)])}
          {"".join([f'{x.code_generation_index_name}, ' for x in sorted(child_attribute.deviceKernel.kernelConnectivity, key = lambda y: y.fullName)])}
          {"".join([f'{x.code_generation_csr_name}, ' for x in child_attribute.deviceKernel.kernelConnectivity if x.dimension == 0])}
          {"".join([f'{x.code_generation_counts_name},' for x in child_attribute.deviceKernel.kernelPrimitiveUnions])}
          {current.correspondance.fullName}_index - {current.fullName}_primitive_total_counts[{current.fullName}_primitive_index],
          {f'&{attribute_name}' if current.size == 1 else attribute_name + ".data()"}
        );
        break;
'''
    code_string += '''
      default:
        break;
    } // close switch
  } // close the local construction
'''
    self.__code_strings.append(code_string)
    return ""


  def __generate_code_for_named_attribute(self, current: ya.attribute) -> str:
    attribute_name, attribute_initialization = self.__generate_attribute_name_and_initialization(current)
    self.__code_strings.append(f'''
  {attribute_initialization};
''')
    if current.deviceKernel is not None:
      self.__code_strings.append(f'''
  {current.fullName}_device_function(
    {"".join([f'{x.code_generation_data_name}, ' for x in current.deviceKernel.kernelDatas])}
    {"".join([f'{x.code_generation_index_name}, ' for x in current.deviceKernel.kernelConnectivity])}
    {"".join([f'{x.code_generation_csr_name}, ' for x in current.deviceKernel.kernelConnectivity if x.dimension == 0])}
    {"".join([f'{x.code_generation_counts_name},' for x in current.deviceKernel.kernelPrimitiveUnions])}
    {"0" if ((current.correspondance.type == "scene" or current.correspondance.type == "mesh") and (self.__input.correspondance.fullName != current.correspondance.fullName)) else f"{current.correspondance.fullName}_index"},
    {f'{attribute_name}.data()' if current.size > 1 else f'&{attribute_name}'}
  );
''')
    else:
      # we found the same kernel
      # we need to replace the kernel calls
      replacement = self.__childrenAttributeKernels[current.hash]
      self.__code_strings.append(f'''
  {current.fullName}_device_function(
    {"".join([f'{x.code_generation_data_name}, ' for x in replacement.deviceKernel.kernelDatas])}
    {"".join([f'{x.code_generation_index_name}, ' for x in replacement.deviceKernel.kernelConnectivity])}
    {"".join([f'{x.code_generation_csr_name}, ' for x in replacement.deviceKernel.kernelConnectivity if x.dimension == 0])}
    {"".join([f'{x.code_generation_counts_name},' for x in replacement.deviceKernel.kernelPrimitiveUnions])}
    {replacement.correspondance.fullName}_index,
    {f'{attribute_name}.data()' if current.size > 1 else f'&{attribute_name}'}
  );
''')
    return ""




#   def __generate_code_for_sum_and_average(self, current: ya.attribute) -> str:
#     # get the children attribute
#     children_attribute = current.children[0]
#     children_attribute_name: str = ""
#     if children_attribute.name == "":
#       children_attribute_name = children_attribute.fullName
#     else:
#       children_attribute_name = children_attribute.fullName
#     self.__code_strings.append(f'''
#   double {current.fullName}_local_data_temp[{current.size}] = {{0}};
# ''')
#     if current.through.dimension == 0:
#       self.__code_strings.append(f'''
#   // grab the starting and ending index since the connectivity is not fixed
#   unsigned int {current.through.fullName}_starting_index = {current.through.code_generation_csr_name}[{current.through.fromPrimitive.fullName}_index];
#   unsigned int {current.through.fullName}_ending_index = {current.through.code_generation_csr_name}[{current.through.fromPrimitive.fullName}_index + 1];
# ''')
#     else:
#       self.__code_strings.append(f'''
#   // we know where to start and end
#   unsigned int {current.through.fullName}_starting_index = {current.through.fromPrimitive.fullName}_index * {current.through.dimension};
#   unsigned int {current.through.fullName}_ending_index = {current.through.fromPrimitive.fullName}_index * {current.through.dimension} + {current.through.dimension};
# ''')
#     self.__code_strings.append(f'''
#   for (unsigned int i = {current.through.fullName}_starting_index; i < {current.through.fullName}_ending_index; i++){{
#     // grab the index for the through attribute
#     unsigned int {current.through.fullName}_index = {current.through.code_generation_index_name}[i];
#     // now for each row, grab the data
#     double {current.fullName}_local_data_row_temp[{children_attribute.size}] = {{0}};
#     {children_attribute_name}_device_function(
#       {"".join([f'{x.code_generation_data_name}, ' for x in sorted(children_attribute.deviceKernel.kernelDatas, key = lambda y: y.fullName)])}
#       {"".join([f'{x.code_generation_index_name}, ' for x in sorted(children_attribute.deviceKernel.kernelConnectivity, key = lambda y: y.fullName)])}
#       {"".join([f'{x.code_generation_csr_name}, ' for x in children_attribute.deviceKernel.kernelConnectivity if x.dimension == 0])},
#       {"".join([f'{x.code_generation_counts_name},' for x in sorted(children_attribute.deviceKernel.kernelPrimitiveUnions, key = lambda y: y.fullName)])}
#       {current.through.fullName}_index,
#       {current.fullName}_local_data_row_temp
#     );

#     // add the data back
#     for (unsigned int j = 0; j < {children_attribute.size}; j++){{
#       {current.fullName}_local_data_temp[j] += {current.fullName}_local_data_row_temp[j];
#     }}
#   }}
# ''')
#     if current.size > 1:
#       if current.rows == 1 or current.cols == 1:
#         self.__code_strings.append(f'''
#     // we now need to put it into the matrix
#     Eigen::Map<Eigen::Matrix<double, {current.rows}, {current.cols}>> {current.fullName}_local_data({current.fullName}_local_data_temp);
#   ''')
#       else:
#         self.__code_strings.append(f'''
#   // we now need to put it into the matrix
#   Eigen::Map<Eigen::Matrix<double, {current.rows}, {current.cols}, Eigen::RowMajor>> {current.fullName}_local_data({current.fullName}_local_data_temp);
# ''')
#     else:
#       self.__code_strings.append(f'''
#   // we put it back to just a single value
#   double {current.fullName}_local_data = {current.fullName}_local_data_temp[0];
# ''')
#     if current.operator == ya.AVERAGE:
#       self.__code_strings.append(f'''
#   // perform average
#   {current.fullName}_local_data /= max({current.through.fullName}_ending_index - {current.through.fullName}_starting_index, static_cast<unsigned int>(1));
# ''')
