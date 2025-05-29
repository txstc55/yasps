# cython: language_level=3
from __future__ import annotations
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
    self.__stack: List[ya.attribute] = list(input.children)
    self.__childrenAttributeKernels: Dict[int, ya.attribute] = {}
    self.__attribute_replacements: Dict[int, Tuple[ya.attribute, int]] = {}
    self.__num_intermediates: int = 0
    self.__code_strings: List[str] = []
    self.__repeated_intermediates: Set[int] = set()
    self.__seen_elements: Set[int] = set()


  def __generateCodeOrderDFS(self, current):
    # print("Generating code for attribute:", current.fullName)
    # print(current)
    if self.__input.deviceKernel is not None:
      # nothing to do
      return
    if current.hash in self.__seen_elements:
      return
    if current.operator == ya.FLOAT or current.operator == ya.INDEX:
      return
    elif current.isFloatMat:
      if current.correspondance.fullName != self.__input.correspondance.fullName:
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
    elif current.correspondance.fullName == self.__input.correspondance.fullName:
      if current.name != "":
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
    if len(self.__stack) == 0 and self.__input.operator == ya.DATA:
      # this attribute is a data, generate special code for it
      current: ya.attribute = self.__input
      kernelString: str = f'''
  #pragma unroll
  for (unsigned int i = 0; i < {current.size}; i++) {{
    result[i] = {current.code_generation_data_name}[{current.correspondance.fullName}_index * {current.size} + i];
  }}'''
      kernelHeader: str = f'''
__device__ void {current.fullName}_device_function(const double* {current.code_generation_data_name}, unsigned int {current.correspondance.fullName}_index, double* result)'''
      current.deviceKernel = deviceKernel(f'{kernelHeader}{{\n{kernelString}\n}}', kernelHeader, [current], [], [], []) # initialize the kernel with the code, the header, self as data, no connectivity, no dependents
      return

    # actually generate the code
    # print(f"At {self.__input.fullName}")
    # print("With children")
    # print(f"Children: {[x.fullName for x in self.__input.children]}")
    for item in self.__input.children:
      self.__generateCodeOrderDFS(item)
    self.__order.append(self.__input)
    # print(f"At order append finish for {self.__input.fullName}")
    # go from bottom to top
    # check how many items appeared more than once
    # order_counts: Dict[int, int] = {}

    # # we check if the item actually needs to be initialized
    # # since some of the elements are only computed once
    # for item in self.__order:
    #   if item.hash in order_counts:
    #     order_counts[item.hash] += 1
    #   else:
    #     order_counts[item.hash] = 1
    for current in self.__order:
      # print("Generating code for")
      # print(str(current))
      # print(current.hash)
      if current.hash in self.__attribute_replacements:
        # we don't need to do anything about it
        pass
      elif current.hash == self.__input.hash or current.name == "":
        # we need to generate the code accordingly
        if current.name != "":
          self.__attribute_replacements[current.hash] = (current, -1)
        else:
          self.__attribute_replacements[current.hash] = (current, self.__num_intermediates)
          self.__num_intermediates += 1

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
          elif current.operator == ya.DATA:
            # the only reason we will reach here
            # is because we are at the root node
            # since data attribute should always have a name
            # but we already handled that case
            return
          elif current.operator == ya.ARRAY:
            self.__generate_code_for_array(current)
          elif current.operator == ya.JOIN:
            self.__generate_code_for_join(current)
          elif current.operator == ya.SUM or current.operator == ya.AVERAGE:
            self.__generate_code_for_sum_and_average(current)
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
        if current.size == 1:
          self.__code_strings.append(f'''
  double {current.fullName}_local_data = 0.0;
''')
        else:
          self.__code_strings.append(f'''
  double {current.fullName}_local_data_temp[{current.size}];
''')
#         if current.correspondance.type == "scene" or current.correspondance.type == "mesh":
#           # we also need to check if the indexing is already a part of the input
#           if self.__input.correspondance.type != "scene" and self.__input.correspondance.type != "mesh":
#             # this is one piece of data, that doesn't really need indexing
#             # we set the index to 0
#             self.__code_strings.append(f'''
# // add 0 indexing since it is a scene or mesh data
# #define {current.correspondance.fullName}_index 0
# ''')
        if current.deviceKernel is not None:
          self.__code_strings.append(f'''
  {current.fullName}_device_function(
  {"".join([f'{x.code_generation_data_name}, ' for x in current.deviceKernel.kernelDatas])}
  {"".join([f'{x.code_generation_index_name}, ' for x in current.deviceKernel.kernelConnectivity])}
  {"".join([f'{x.code_generation_csr_name}, ' for x in current.deviceKernel.kernelConnectivity if x.dimension == 0])}
  {"".join([f'{x.code_generation_counts_name},' for x in current.deviceKernel.kernelPrimitiveUnions])}
  {"0" if ((current.correspondance.type == "scene" or current.correspondance.type == "mesh") and (self.__input.correspondance.type != "scene" and self.__input.correspondance.type != "mesh")) else f"{current.correspondance.fullName}_index"},
  {f'{current.fullName}_local_data_temp' if current.size > 1 else f'&{current.fullName}_local_data'});
  ''')
        else:
          # we found the same kernel
          # we need to replace the kernel calls
          # if str(current.hash) == "73073881234865546943141648636381669944003170022450331731243882876784723308431":
          #   print("Found bad hash")
          #   print(str(current))
          replacement = self.__childrenAttributeKernels[current.hash]
          self.__code_strings.append(f'''
  {current.fullName}_device_function(
    {"".join([f'{x.code_generation_data_name}, ' for x in replacement.deviceKernel.kernelDatas])}
    {"".join([f'{x.code_generation_index_name}, ' for x in replacement.deviceKernel.kernelConnectivity])}
    {"".join([f'{x.code_generation_csr_name}, ' for x in replacement.deviceKernel.kernelConnectivity if x.dimension == 0])}
    {"".join([f'{x.code_generation_counts_name},' for x in replacement.deviceKernel.kernelPrimitiveUnions])}
    {replacement.correspondance.fullName}_index,
    {f'{current.fullName}_local_data_temp' if current.size > 1 else f'&{current.fullName}_local_data'});
  ''')
        if current.size > 1:
          if current.rows == 1 or current.cols == 1:
            self.__code_strings.append(f'''
  Eigen::Map<Eigen::Matrix<double, {current.rows}, {current.cols}>> {current.fullName}_local_data({current.fullName}_local_data_temp);
''')
          else:
            self.__code_strings.append(f'''
  Eigen::Map<Eigen::Matrix<double, {current.rows}, {current.cols}, Eigen::RowMajor>> {current.fullName}_local_data({current.fullName}_local_data_temp);
''')
        # add to intermediate names
        self.__attribute_replacements[current.hash] = (current, -1)

    # now we are done with the computation
    # need to put back the result
    attributeName: str = ""
    if self.__input.name == "":
      attributeName = self.__input.fullName
    else:
      attributeName = self.__input.fullName
    # we have finished the computation, add a line to store the result
    if self.__input.size == 1:
      self.__code_strings.append(f'''
  result[0] = {self.getIntermediateName(self.__input)};
''')
    else:
      self.__code_strings.append(f'''
  // put the result back
  Eigen::Matrix<double, {self.__input.rows}, {self.__input.cols}{'' if self.__input.rows == 1 or self.__input.cols == 1 else ', Eigen::RowMajor'}> {self.getIntermediateName(self.__input)}_materialized({self.getIntermediateName(self.__input)});
  #pragma unroll
  for (unsigned int i = 0; i < {self.__input.size}; i++){{
    result[i] = {self.getIntermediateName(self.__input)}_materialized.data()[i];
  }}
''')
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
    self.__code_strings = [x.strip('\n') for x in self.__code_strings]
    kernelString: str = "\n".join(self.__code_strings)

    # now we generate the device kernel
    self.__input.deviceKernel = deviceKernel(f'{headerString}{{\n{kernelString}\n}}', headerString, allDatas, allConnectivities, allPrimitiveUnions, allDependencies)
    # print(f"Device kernel generated for {self.__input.fullName}")
    # print(f"All intermediate count: {len(self.__attribute_replacements)}")
    # print(f"num intermediates: {self.__num_intermediates}")


  # get the name of the intermediate variables
  def getIntermediateName(self, attribute: ya.attribute) -> str:
    if attribute.operator == ya.FLOAT:
      return str(attribute.float_value)
    # return the name of the intermediate value
    attribute_hash = attribute.hash
    if attribute_hash not in self.__attribute_replacements:
      raise ValueError(f"codeGenerator.getIntermediateName: attribute hash not found in self.__attribute_replacements. {str(attribute)} hash is: {attribute_hash}")
    if self.__attribute_replacements[attribute_hash][1] == -1:
      return f"{self.__attribute_replacements[attribute_hash][0].fullName}_local_data"
    else:
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



  def __generate_attribute_name_and_initialization(self, current: ya.attribute) -> Tuple[str, str]:
    # we need to generate the code accordingly
    attribute_initialization: str = ""
    attribute_name: str = ""
    if current.name != "":
      attribute_name = current.fullName + "_local_data"
      self.__attribute_replacements[current.hash] = (current, -1)
    else:
      ind: int = self.__attribute_replacements[current.hash][1]
      attribute_name = f"INTERMEDIATE_{ind}"

    if current.size == 1:
      attribute_initialization = f"double {attribute_name}"
    else:
      if current.rows == 1 or current.cols == 1:
        attribute_initialization = f"Eigen::Matrix<double, {current.rows}, {current.cols}> {attribute_name}"
      else:
        attribute_initialization = f"Eigen::Matrix<double, {current.rows}, {current.cols}, Eigen::RowMajor> {attribute_name}"
    return attribute_name, attribute_initialization


  def __generate_code_for_type_0(self, current: ya.attribute) -> None:
    attribute_name, attribute_initialization = self.__generate_attribute_name_and_initialization(current)
#     if current.hash not in self.__repeated_intermediates and current.hash != self.__input.hash:
#       if current.size == 1:
#         self.__code_strings.append(f'''
# # define {attribute_name} ({current.operator.name}({self.getIntermediateName(current.children[0])}))''')
#       else:
#         self.__code_strings.append(f'''
# # define {attribute_name} ({self.getIntermediateName(current.children[0])}.array().{current.operator.name}())''')
#       return
    # different code generation for scalar and double
    if current.size == 1:
      self.__code_strings.append(f'''
  double {attribute_name} = {current.operator.name}({self.getIntermediateName(current.children[0])});''')
    else:
      self.__code_strings.append(f'''
  // allocate the space since this operation is most likely going to be expansive
  {attribute_initialization} = {self.getIntermediateName(current.children[0])}.array().{current.operator.name}();''')


  def __generate_code_for_type_1(self, current: ya.attribute) -> None:
    attribute_name, attribute_initialization = self.__generate_attribute_name_and_initialization(current)
#     if current.hash not in self.__repeated_intermediates and current.hash != self.__input.hash:
#       self.__code_strings.append(f'''
# # define {attribute_name} ({self.getIntermediateName(current.children[0])} {current.operator.name} {self.getIntermediateName(current.children[1])})''')
#       return
    if current.operator == ya.MUL or current.operator == ya.DIV:
      self.__code_strings.append(f'''
  {attribute_initialization} = {self.getIntermediateName(current.children[0])} {current.operator.name} {self.getIntermediateName(current.children[1])};''')
    else:
      self.__code_strings.append(f'''
  auto {attribute_name} = {self.getIntermediateName(current.children[0])} {current.operator.name} {self.getIntermediateName(current.children[1])};''')


  def __generate_code_for_type_2(self, current: ya.attribute) -> None:
    attribute_name, attribute_initialization = self.__generate_attribute_name_and_initialization(current)
    # for type 2 there isn't many operators
    # currently we have select or power
    # and power is forbidden on matrix
    # so we can always do it this way as op(a, b, c, d, ...)
    if current.operator == ya.SELECT:
      self.__code_strings.append(f'''
  {attribute_initialization} = {self.getIntermediateName(current.children[0])} ? {self.getIntermediateName(current.children[1])} : {self.getIntermediateName(current.children[2])};''')

    else:
      self.__code_strings.append(f'''
  {attribute_initialization} = {current.operator.name}({", ".join([self.getIntermediateName(x) for x in current.children])});''')


  def __generate_code_for_index(self, current: ya.attribute) -> None:
    # this should never ever happen
    raise ValueError("codeGenerator.generateCode: INDEX operator should never be reached.")

  def __generate_code_for_neg(self, current: ya.attribute) -> None:
    attribute_name, attribute_initialization = self.__generate_attribute_name_and_initialization(current)
    self.__code_strings.append(f'''
    {attribute_initialization} = -{self.getIntermediateName(current.children[0])};''')

  def __generate_code_for_float(self, current: ya.attribute) -> None:
    # the float attribute can only happen if it is a root node
    # because when traversing we never put it on the stack
    self.__code_strings.append(f'''
  result[0] = {current.float_value};''')

  def __generate_code_for_array_access(self, current: ya.attribute) -> None:
    attribute_name, attribute_initialization = self.__generate_attribute_name_and_initialization(current)
    # generate code for array access
    index_value = current.children[1].index_value
    row_num = index_value // current.children[0].cols
    col_num = index_value % current.children[0].cols
    child_rows = current.children[0].rows
    child_cols = current.children[0].cols
    if child_rows == 1 and child_cols == 1:
      # this is a direct access to a scalar
      self.__code_strings.append(f'''
  {attribute_initialization} = {self.getIntermediateName(current.children[0])};''')
    else:
      self.__code_strings.append(f'''
  {attribute_initialization} = {self.getIntermediateName(current.children[0])}({row_num}, {col_num});''')

  def __generate_code_for_array(self, current: ya.attribute) -> None:
    attribute_name, attribute_initialization = self.__generate_attribute_name_and_initialization(current)
    # generate code for array
    self.__code_strings.append(f'''
  {attribute_initialization};
  {attribute_name} << {", ".join([self.getIntermediateName(x) for x in current.children])};
''')


  def __generate_code_for_join(self, current: ya.attribute) -> None:
    # need to generate the code for the joining operation
    # we know the children must be a named attribute for the joining operator
    children_attribute = current.children[0] # should only have one child
    children_attribute_name: str = ""
    if children_attribute.name == "":
      children_attribute_name = children_attribute.fullName
    else:
      children_attribute_name = children_attribute.fullName
    self.__code_strings.append(f'''
  double {current.fullName}_local_data_temp[{current.size}];
  for (unsigned int i = 0; i < {current.through.dimension}; i++){{
  // grab the index for the through attribute
    unsigned int {current.through.fullName}_index = {current.through.code_generation_index_name}[{current.through.fromPrimitive.fullName}_index * {current.through.dimension} + i];
    // now for each row, grab the data
    double {current.fullName}_local_data_row_temp[{children_attribute.size}];
    {children_attribute_name}_device_function(
      {"".join([f'{x.code_generation_data_name}, ' for x in sorted(children_attribute.deviceKernel.kernelDatas, key = lambda y: y.fullName)])}
      {"".join([f'{x.code_generation_index_name}, ' for x in sorted(children_attribute.deviceKernel.kernelConnectivity, key = lambda y: y.fullName)])}
      {"".join([f'{x.code_generation_csr_name}, ' for x in children_attribute.deviceKernel.kernelConnectivity if x.dimension == 0])}
      {"".join([f'{x.code_generation_counts_name},' for x in sorted(children_attribute.deviceKernel.kernelPrimitiveUnions, key = lambda y: y.fullName)])}
      {current.through.fullName}_index,
      {current.fullName}_local_data_row_temp
    );
    #pragma unroll
    for (unsigned int j = 0; j < {children_attribute.size}; j++){{ // copy the data
      {current.fullName}_local_data_temp[i * {int(children_attribute.size)} + j] = {current.fullName}_local_data_row_temp[j];
    }}
  }}
''')
    # put it back according to size
    if current.size > 1:
      if current.rows == 1 or current.cols == 1:
        self.__code_strings.append(f'''
    // we now need to put it into the matrix
    Eigen::Map<Eigen::Matrix<double, {current.rows}, {current.cols}>> {current.fullName}_local_data({current.fullName}_local_data_temp);
  ''')
      else:
        self.__code_strings.append(f'''
  // we now need to put it into the matrix
  Eigen::Map<Eigen::Matrix<double, {current.rows}, {current.cols}, Eigen::RowMajor>> {current.fullName}_local_data({current.fullName}_local_data_temp);
''')
    else:
      self.__code_strings.append(f'''
  // we put it back to just a single value
  double {current.fullName}_local_data = {current.fullName}_local_data_temp[0];
''')

  def __generate_code_for_sum_and_average(self, current: ya.attribute) -> None:
    # get the children attribute
    children_attribute = current.children[0]
    children_attribute_name: str = ""
    if children_attribute.name == "":
      children_attribute_name = children_attribute.fullName
    else:
      children_attribute_name = children_attribute.fullName
    self.__code_strings.append(f'''
  double {current.fullName}_local_data_temp[{current.size}] = {{0}};
''')
    if current.through.dimension == 0:
      self.__code_strings.append(f'''
  // grab the starting and ending index since the connectivity is not fixed
  unsigned int {current.through.fullName}_starting_index = {current.through.code_generation_csr_name}[{current.through.fromPrimitive.fullName}_index];
  unsigned int {current.through.fullName}_ending_index = {current.through.code_generation_csr_name}[{current.through.fromPrimitive.fullName}_index + 1];
''')
    else:
      self.__code_strings.append(f'''
  // we know where to start and end
  unsigned int {current.through.fullName}_starting_index = {current.through.fromPrimitive.fullName}_index * {current.through.dimension};
  unsigned int {current.through.fullName}_ending_index = {current.through.fromPrimitive.fullName}_index * {current.through.dimension} + {current.through.dimension};
''')
    self.__code_strings.append(f'''
  for (unsigned int i = {current.through.fullName}_starting_index; i < {current.through.fullName}_ending_index; i++){{
    // grab the index for the through attribute
    unsigned int {current.through.fullName}_index = {current.through.code_generation_index_name}[i];
    // now for each row, grab the data
    double {current.fullName}_local_data_row_temp[{children_attribute.size}];
    {children_attribute_name}_device_function(
      {"".join([f'{x.code_generation_data_name}, ' for x in sorted(children_attribute.deviceKernel.kernelDatas, key = lambda y: y.fullName)])}
      {"".join([f'{x.code_generation_index_name}, ' for x in sorted(children_attribute.deviceKernel.kernelConnectivity, key = lambda y: y.fullName)])}
      {"".join([f'{x.code_generation_csr_name}, ' for x in children_attribute.deviceKernel.kernelConnectivity if x.dimension == 0])},
      {"".join([f'{x.code_generation_counts_name},' for x in sorted(children_attribute.deviceKernel.kernelPrimitiveUnions, key = lambda y: y.fullName)])}
      {current.through.fullName}_index,
      {current.fullName}_local_data_row_temp
    );

    // add the data back
    for (unsigned int j = 0; j < {children_attribute.size}; j++){{
      {current.fullName}_local_data_temp[j] += {current.fullName}_local_data_row_temp[j];
    }}
  }}
''')
    if current.size > 1:
      if current.rows == 1 or current.cols == 1:
        self.__code_strings.append(f'''
    // we now need to put it into the matrix
    Eigen::Map<Eigen::Matrix<double, {current.rows}, {current.cols}>> {current.fullName}_local_data({current.fullName}_local_data_temp);
  ''')
      else:
        self.__code_strings.append(f'''
  // we now need to put it into the matrix
  Eigen::Map<Eigen::Matrix<double, {current.rows}, {current.cols}, Eigen::RowMajor>> {current.fullName}_local_data({current.fullName}_local_data_temp);
''')
    else:
      self.__code_strings.append(f'''
  // we put it back to just a single value
  double {current.fullName}_local_data = {current.fullName}_local_data_temp[0];
''')
    if current.operator == ya.AVERAGE:
      self.__code_strings.append(f'''
  // perform average
  {current.fullName}_local_data /= max({current.through.fullName}_ending_index - {current.through.fullName}_starting_index, static_cast<unsigned int>(1));
''')


  def __generate_code_for_transpose(self, current: ya.attribute) -> None:
    attribute_name, _ = self.__generate_attribute_name_and_initialization(current)
    self.__code_strings.append(f'''
  // make transpose expression rather than a copy
  auto {attribute_name} = {self.getIntermediateName(current.children[0])}.transpose();''')

  def __generate_code_for_broadcast_add(self, current: ya.attribute) -> None:
    attribute_name, _ = self.__generate_attribute_name_and_initialization(current)
    self.__code_strings.append(f'''
  // broadcast addition
  auto {attribute_name} = {self.getIntermediateName(current.children[0])}.array() + {self.getIntermediateName(current.children[1])};''')

  def __generate_code_for_broadcast_sub(self, current: ya.attribute) -> None:
    attribute_name, _ = self.__generate_attribute_name_and_initialization(current)
    self.__code_strings.append(f'''
  // broadcast subtraction
  auto {attribute_name} = {self.getIntermediateName(current.children[0])}.array() - {self.getIntermediateName(current.children[1])};''')

  def __generate_code_for_row(self, current: ya.attribute) -> None:
    attribute_name, _ = self.__generate_attribute_name_and_initialization(current)
    self.__code_strings.append(f'''
  // getting row by expression template
  auto {attribute_name} = {self.getIntermediateName(current.children[0])}.row({current.children[1].index_value});''')

  def __generate_code_for_col(self, current: ya.attribute) -> None:
    attribute_name, _ = self.__generate_attribute_name_and_initialization(current)
    self.__code_strings.append(f'''
  // getting column by expression template
  auto {attribute_name} = {self.getIntermediateName(current.children[0])}.col({current.children[1].index_value});''')

  def __generate_code_for_cross(self, current: ya.attribute) -> None:
    attribute_name, attribute_initialization = self.__generate_attribute_name_and_initialization(current)
    self.__code_strings.append(f'''
  {attribute_initialization} = {self.getIntermediateName(current.children[0])}.cross({self.getIntermediateName(current.children[1])});''')

  def __generate_code_for_norm(self, current: ya.attribute) -> None:
    attribute_name, attribute_initialization = self.__generate_attribute_name_and_initialization(current)
    self.__code_strings.append(f'''
  {attribute_initialization} = {self.getIntermediateName(current.children[0])}.norm();''')

  def __generate_code_for_det(self, current: ya.attribute) -> None:
    attribute_name, attribute_initialization = self.__generate_attribute_name_and_initialization(current)
    self.__code_strings.append(f'''
  {attribute_initialization} = {self.getIntermediateName(current.children[0])}.determinant();''')

  def __generate_code_for_inverse(self, current: ya.attribute) -> None:
    attribute_name, attribute_initialization = self.__generate_attribute_name_and_initialization(current)
    self.__code_strings.append(f'''
  {attribute_initialization} = {self.getIntermediateName(current.children[0])}.inverse();''')

  def __generate_code_for_dot(self, current: ya.attribute) -> None:
    attribute_name, attribute_initialization = self.__generate_attribute_name_and_initialization(current)
    self.__code_strings.append(f'''
  {attribute_initialization} = {self.getIntermediateName(current.children[0])}.dot({self.getIntermediateName(current.children[1])});''')

  def __generate_code_for_resize(self, current: ya.attribute) -> None:
    # we need to generate the code accordingly
    origin_mat = current.children[0]
    attribute_initialization: str = ""
    attribute_name: str = ""
    if current.name != "":
      attribute_name = current.fullName + "_local_data"
      self.__attribute_replacements[current.hash] = (current, -1)

    else:
      ind: int = self.__attribute_replacements[current.hash][1]
      attribute_name = f"INTERMEDIATE_{ind}"

    if origin_mat.rows == 1 or origin_mat.cols == 1:
      attribute_initialization = f"Eigen::Matrix<double, {origin_mat.rows}, {origin_mat.cols}> {attribute_name}_before_resize = {self.getIntermediateName(current.children[0])};\n  "
    else:
      attribute_initialization = f"Eigen::Matrix<double, {origin_mat.rows}, {origin_mat.cols}, Eigen::RowMajor> {attribute_name}_before_resize = {self.getIntermediateName(current.children[0])};\n  "
    if current.rows == 1 or current.cols == 1:
      attribute_initialization += f'''Eigen::Matrix<double, {current.rows}, {current.cols}> {attribute_name}'''
    else:
      attribute_initialization += f'''Eigen::Matrix<double, {current.rows}, {current.cols}, Eigen::RowMajor> {attribute_name}'''
    # copy the data and resize
    self.__code_strings.append(f'''
  {attribute_initialization}({attribute_name}_before_resize.data());''')

  def __generate_code_for_spd(self, current: ya.attribute) -> None:
    attribute_name, attribute_initialization = self.__generate_attribute_name_and_initialization(current)
    if current.size == 1:
      spd_method = current.children[1].index_value
      if spd_method == 1:
        self.__code_strings.append(f'''
  {attribute_initialization} = abs({self.getIntermediateName(current.children[0])});''')
      elif spd_method == 2:
        self.__code_strings.append(f'''
  {attribute_initialization} = max(0.0, {self.getIntermediateName(current.children[0])});''')
    else:
      if current.rows <= 3:
        self.__code_strings.append(f'''
    {attribute_initialization} = Eigen::Matrix<double, {current.rows}, {current.cols}, Eigen::RowMajor>::Zero();
    spd_projection_small<{current.children[0].rows}>({self.getIntermediateName(current.children[0])}.data(), {attribute_name}.data(), {current.children[1].index_value});
  ''')
      else:
        # we do the normal projection
        self.__code_strings.append(f'''
  {attribute_initialization} = Eigen::Matrix<double, {current.rows}, {current.cols}, Eigen::RowMajor>::Zero();
  spd_projection<{current.children[0].rows}>({self.getIntermediateName(current.children[0])}.data(), {attribute_name}.data(), {current.children[1].index_value});
''')

  def __generate_code_for_union(self, current: ya.attribute) -> None:
    # need to generate the code for the joining operation
    # we know the children must be a named attribute for the joining operator
    # print("We are now generating code for union")
    code_string = ""
    code_string += f'''
  double {current.fullName}_local_data_temp[{current.size}] = {{0}};
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
    # print("Before for loop")
    for i in range(len(current.children)):
      child_attribute = current.children[i]
      code_string += f'''
      case {i}:
        {current.children[i].fullName}_device_function(
          {"".join([f'{x.code_generation_data_name}, ' for x in sorted(child_attribute.deviceKernel.kernelDatas, key = lambda y: y.fullName)])}
          {"".join([f'{x.code_generation_index_name}, ' for x in sorted(child_attribute.deviceKernel.kernelConnectivity, key = lambda y: y.fullName)])}
          {"".join([f'{x.code_generation_csr_name}, ' for x in child_attribute.deviceKernel.kernelConnectivity if x.dimension == 0])}
          {"".join([f'{x.code_generation_counts_name},' for x in child_attribute.deviceKernel.kernelPrimitiveUnions])}
          {current.correspondance.fullName}_index - {current.fullName}_primitive_total_counts[{current.fullName}_primitive_index],
          {current.fullName}_local_data_temp
        );
        break;
'''
    # print("After for loop")
    code_string += '''
      default:
        break;
    } // close switch
  } // close the local construction
'''
    # put it back according to size
    if current.size > 1:
      if current.rows == 1 or current.cols == 1:
        code_string += f'''
  // we now need to put it into the matrix
  Eigen::Map<Eigen::Matrix<double, {current.rows}, {current.cols}>> {current.fullName}_local_data({current.fullName}_local_data_temp);
'''
      else:
        code_string += f'''
  // we now need to put it into the matrix
  Eigen::Map<Eigen::Matrix<double, {current.rows}, {current.cols}, Eigen::RowMajor>> {current.fullName}_local_data({current.fullName}_local_data_temp);
'''
    else:
      # it is a singular scalar data
      code_string += f'''
    // we put it back to just a single value
  double {current.fullName}_local_data = {current.fullName}_local_data_temp[0];
'''
    self.__code_strings.append(code_string)
