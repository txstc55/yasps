# cython: language_level=3
from __future__ import annotations
import yasps.attribute as ya
from yasps.connectivity import connectivity
from typing import Dict, List, Set
from yasps.deviceKernel import deviceKernel
from yasps.globalKernel import globalKernel

class codeGenerator:
  # generate the code for attribute
  def __init__(self, input: ya.attribute):
    self.__input: ya.attribute = input
    self.__order: List[ya.attribute] = [input]
    self.__stack: List[ya.attribute] = list(input.children)
    self.__childrenAttributeKernels: Dict[int, ya.attribute] = {}
    self.__attribute_replacements: Dict[int, int] = {}
    self.__num_intermediates: int = 0
    self.__code_strings: List[str] = []

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
            childCodeGenerator = codeGenerator(current)
            childCodeGenerator.generateCode()
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
              childCodeGenerator = codeGenerator(current)
              childCodeGenerator.generateCode()
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
              childCodeGenerator = codeGenerator(current)
              childCodeGenerator.generateCode()
              self.__childrenAttributeKernels[current.hash] = current
          else:
            # this should not happen, raise an error
            raise ValueError("codeGenerator.__generateCodeOrder: actually going to a child without a name and not the same correspondance.")


  def generateCodeOrder(self) -> None:
    self.__generateCodeOrder()


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
__device__ __inline__ void {current.fullName}_device_function(const double* {current.code_generation_data_name}, unsigned int {current.correspondance.fullName}_index, double* result)'''
      current.deviceKernel = deviceKernel(f'{kernelHeader}{{\n{kernelString}\n}}', kernelHeader, [current], [], []) # initialize the kernel with the code, the header, self as data, no connectivity, no dependents
      return

    # actually generate the code
    self.__generateCodeOrder()
    # go from bottom to top
    for current in self.__order[::-1]:
      if current.hash in self.__attribute_replacements:
        # we don't need to do anything about it
        pass
      elif current.hash == self.__input.hash or current.name == "":
        # we need to generate the code accordingly
        attribute_initialization: str = ""
        attribute_name: str = ""
        if current.name != "":
          attribute_name = current.fullName + "_local_data"
          self.__attribute_replacements[current.hash] = -1
        else:
          attribute_name = f"INTERMEDIATE_{self.__num_intermediates}"
          self.__attribute_replacements[current.hash] = self.__num_intermediates
          self.__num_intermediates += 1

        if current.size == 1:
          attribute_initialization = f"double {attribute_name}"
        else:
          if current.rows == 1 or current.cols == 1:
            attribute_initialization = f"Eigen::Matrix<double, {current.rows}, {current.cols}> {attribute_name}"
          else:
            attribute_initialization = f"Eigen::Matrix<double, {current.rows}, {current.cols}, Eigen::RowMajor> {attribute_name}"

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
          elif current.operator == ya.GATHER:
            self.__generate_code_for_gather(current)
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
        if current.correspondance.type == "scene" or current.correspondance.type == "mesh":
          # this is one piece of data, that doesn't really need indexing
          # we set the index to 0
          self.__code_strings.append(f'''
  // add 0 indexing since it is a scene or mesh data
  unsigned int {current.correspondance.fullName}_index = 0;
''')
        self.__code_strings.append(f'''
  {current.fullName}_device_function({"".join([f'{x.code_generation_data_name}, ' for x in current.deviceKernel.kernelDatas])}{"".join([f'{x.code_generation_index_name}, ' for x in current.deviceKernel.kernelConnectivity])}{"".join([f'{x.code_generation_csr_name}, ' for x in current.deviceKernel.kernelConnectivity if x.dimension == 0])}{current.correspondance.fullName}_index, {f'{current.fullName}_local_data_temp' if current.size > 1 else f'&{current.fullName}_local_data'});
''')
        if current.size > 1:
          self.__code_strings.append(f'''
  Eigen::Map<Eigen::Matrix<double, {current.rows}, {current.cols}, Eigen::RowMajor>> {current.fullName}_local_data({current.fullName}_local_data_temp);
''')
        # add to intermediate names
        self.__attribute_replacements[current.hash] = -1

    # now we are done with the computation
    # need to put back the result
    attributeName: str = ""
    if self.__input.name == "":
      attributeName = f'attr_{self.__input.hash}'
    else:
      attributeName = self.__input.fullName
    # we have finished the computation, add a line to store the result
    self.__code_strings.append(f'''
  // put the result back
  #pragma unroll
  for (unsigned int i = 0; i < {self.__input.size}; i++){{
    result[i] = {self.getIntermediateName(self.__input)}{"" if self.__input.size == 1 else ".data()[i]"};
  }}
''')
    # now we need to get the datas for generating this kernel
    allNamedAttributeChildren = self.__childrenAttributeKernels.values()
    # get the datas they need
    allDatas: List[ya.attribute] = [item for x in allNamedAttributeChildren for item in x.deviceKernel.kernelDatas] # get all datas
    allConnectivities: List[connectivity] = [item for x in allNamedAttributeChildren for item in x.deviceKernel.kernelConnectivity] # get all the connectivities
    allDependencies: List[deviceKernel] = [item for x in allNamedAttributeChildren for item in x.deviceKernel.dependents] # get all the dependencies as strings
    allDependencies = allDependencies + [x.deviceKernel for x in allNamedAttributeChildren] # also add the children as dependencies

    # if we are a gathering operation
    # we need to set the connectivity
    if self.__input.operator == ya.GATHER or self.__input.operator == ya.SUM or self.__input.operator == ya.AVERAGE:
      allConnectivities.append(self.__input.through)

    # sort and remove duplicates
    allDatas = sorted(set(allDatas), key = lambda x: x.fullName)
    allConnectivities = sorted(set(allConnectivities), key = lambda x: x.fullName)
    allDependencies = sorted(set(allDependencies), key = lambda x: x.kernelHeader)

    # now we generate header
    headerString: str = f'''
__device__ __inline__ void {attributeName}_device_function({"".join([f"const double* {x.code_generation_data_name}, " for x in allDatas])}{"".join([f"const unsigned int* {x.code_generation_index_name}, " for x in allConnectivities])}{"".join([f"const unsigned int* {x.code_generation_csr_name}, " for x in allConnectivities if x.dimension == 0])}unsigned int {self.__input.correspondance.fullName}_index, double* result)'''

    kernelString: str = "\n".join(self.__code_strings)

    # now we generate the device kernel
    self.__input.deviceKernel = deviceKernel(f'{headerString}{{\n{kernelString}\n}}', headerString, allDatas, allConnectivities, allDependencies)


  # get the name of the intermediate variables
  def getIntermediateName(self, attribute: ya.attribute) -> str:
    if attribute.operator == ya.FLOAT:
      return str(attribute.float_value)
    # return the name of the intermediate value
    attribute_hash = attribute.hash
    if attribute_hash not in self.__attribute_replacements:
      raise ValueError("codeGenerator.getIntermediateName: attribute hash not found in self.__attribute_replacements.", str(attribute))
    if self.__attribute_replacements[attribute_hash] == -1:
      return f"{attribute.fullName}_local_data"
    else:
      return f"INTERMEDIATE_{self.__attribute_replacements[attribute_hash]}"






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
      self.__attribute_replacements[current.hash] = -1
    else:
      attribute_name = f"INTERMEDIATE_{self.__num_intermediates}"
      self.__attribute_replacements[current.hash] = self.__num_intermediates
      self.__num_intermediates += 1

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
    if current.operator == ya.MUL or current.operator == ya.DIV:
      self.__code_strings.append(f'''
  // division and multiplication is expansive, allocate some space
  {attribute_initialization} = {self.getIntermediateName(current.children[0])} {current.operator.name} {self.getIntermediateName(current.children[1])};''')
    else:
      self.__code_strings.append(f'''
  // operation is not too expansive, use expression template if possible
  auto {attribute_name} = {self.getIntermediateName(current.children[0])} {current.operator.name} {self.getIntermediateName(current.children[1])};''')


  def __generate_code_for_type_2(self, current: ya.attribute) -> None:
    attribute_name, attribute_initialization = self.__generate_attribute_name_and_initialization(current)
    # for type 2 there isn't many operators
    # currently we have select or power
    # and power is forbidden on matrix
    # so we can always do it this way as op(a, b, c, d, ...)
    self.__code_strings.append(f'''
{attribute_initialization} = {current.operator.name}({", ".join([self.getIntermediateName(x) for x in current.children])});''')


  def __generate_code_for_index(self, current: ya.attribute) -> None:
    # this should never ever happen
    raise ValueError("codeGenerator.generateCode: INDEX operator should never be reached.")

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
    self.__code_strings.append(f'''
{attribute_initialization} = {self.getIntermediateName(current.children[0])}({row_num}, {col_num});''')

  def __generate_code_for_array(self, current: ya.attribute) -> None:
    attribute_name, attribute_initialization = self.__generate_attribute_name_and_initialization(current)
    # generate code for array
    self.__code_strings.append(f'''
  {attribute_initialization};
  {attribute_name} << {", ".join([self.getIntermediateName(x) for x in current.children])};
''')


  def __generate_code_for_gather(self, current: ya.attribute) -> None:
    # need to generate the code for the gathering operation
    # we know the children must be a named attribute for the gathering operator
    children_attribute = current.children[0] # should only have one child
    self.__code_strings.append(f'''
  double {current.fullName}_local_data_temp[{current.size}];
  for (unsigned int i = 0; i < {current.through.dimension}; i++){{
  // grab the index for the through attribute
    unsigned int {current.through.fullName}_index = {current.through.code_generation_index_name}[{current.through.fromPrimitive.fullName}_index * {current.through.dimension} + i];
    // now for each row, grab the data
    double {current.fullName}_local_data_row_temp[{children_attribute.size}];
    {children_attribute.fullName}_device_function({"".join([f'{x.code_generation_data_name}, ' for x in sorted(children_attribute.deviceKernel.kernelDatas, key = lambda y: y.fullName)])}{"".join([f'{x.code_generation_index_name}, ' for x in sorted(children_attribute.deviceKernel.kernelConnectivity, key = lambda y: y.fullName)])}{"".join([f'{x.code_generation_csr_name}, ' for x in children_attribute.deviceKernel.kernelConnectivity if x.dimension == 0])}{current.through.fullName}_index, {current.fullName}_local_data_row_temp);
    #pragma unroll
    for (unsigned int j = 0; j < {children_attribute.size}; j++){{ // copy the data
      {current.fullName}_local_data_temp[i * {int(children_attribute.size)} + j] = {current.fullName}_local_data_row_temp[j];
    }}
  }}
''')
    # put it back according to size
    if current.size > 1:
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
    {children_attribute.fullName}_device_function({"".join([f'{x.code_generation_data_name}, ' for x in sorted(children_attribute.deviceKernel.kernelDatas, key = lambda y: y.fullName)])}{"".join([f'{x.code_generation_index_name}, ' for x in sorted(children_attribute.deviceKernel.kernelConnectivity, key = lambda y: y.fullName)])}{"".join([f'{x.code_generation_csr_name}, ' for x in children_attribute.deviceKernel.kernelConnectivity if x.dimension == 0])}{current.through.fullName}_index, {current.fullName}_local_data_row_temp);

    // add the data back
    for (unsigned int j = 0; j < {children_attribute.size}; j++){{
      {current.fullName}_local_data_temp[j] += {current.fullName}_local_data_row_temp[j];
    }}
  }}
''')
    if current.size > 1:
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
