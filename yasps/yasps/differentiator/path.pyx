from yasps.attribute import attribute
from typing import List, Dict, Set, Tuple
from yasps.attribute import DATA, CONSTANT, JOIN, UNION
class path:
  def __init__(self, global_targets: List[attribute], local_targets: List[attribute] = []):
    self.__path_dict: Dict[attribute, List[attribute]] = {}
    self.__unioned_child_to_its_children: Dict[attribute, List[attribute]] = {}
    self.__paths: List[List[attribute]] = []
    self.__root = None
    self.__global_targets = global_targets
    self.__local_targets = local_targets
    self.__wrt_start_indices: List[int] = []
    self.__compute_wrt_start_indices() # compute how the gradient is placed
    self.__wrt = global_targets if (len(local_targets) == 0) else local_targets

  def __compute_wrt_start_indices(self):
    for item in self.__global_targets:
      if item.isDynamic:
        # for wrt let's disallow dynamic attributes
        raise ValueError("hessian.__compute_wrt_start_indices: wrt is a dynamic attributes.")

    gradient_sizes = [item.size * item.correspondance.numInstances for item in self.__global_targets]
    gradient_segment_start = [0]
    for i in range(1, len(gradient_sizes)):
      gradient_segment_start.append(gradient_segment_start[i - 1] + gradient_sizes[i - 1])
    gradient_segment_start.append(gradient_segment_start[-1] + gradient_sizes[-1])
    self.__wrt_start_indices = gradient_segment_start



  @property
  def path_dict(self) -> Dict[attribute, List[attribute]]:
    """Path roots reachable for each requested attribute."""
    return self.__path_dict

  @path_dict.setter
  def path_dict(self, value: Dict[attribute, List[attribute]]) -> None:
    if not isinstance(value, dict):
      raise TypeError("path.path_dict: value must be a dict.")
    for key, item in value.items():
      if not isinstance(key, attribute):
        raise TypeError("path.path_dict: keys must be yasps.attribute.attribute.")
      if not isinstance(item, list) or any(not isinstance(v, attribute) for v in item):
        raise TypeError("path.path_dict: values must be List[attribute].")
    self.__path_dict = value




  @property
  def unioned_child_to_its_children(self) -> Dict[attribute, List[attribute]]:
    """Mapping from a union child to its expanded list of children."""
    return self.__unioned_child_to_its_children

  @unioned_child_to_its_children.setter
  def unioned_child_to_its_children(self, value: Dict[attribute, List[attribute]]) -> None:
    if not isinstance(value, dict):
      raise TypeError("path.unioned_child_to_its_children: value must be a dict.")
    for key, item in value.items():
      if not isinstance(key, attribute):
        raise TypeError("path.unioned_child_to_its_children: keys must be yasps.attribute.attribute.")
      if not isinstance(item, list) or any(not isinstance(v, attribute) for v in item):
        raise TypeError("path.unioned_child_to_its_children: values must be List[attribute].")
    self.__unioned_child_to_its_children = value

  @property
  def paths(self) -> List[List[attribute]]:
    """Computed list of paths from roots to leaves."""
    return self.__paths

  @paths.setter
  def paths(self, value: List[List[attribute]]) -> None:
    if not isinstance(value, list):
      raise TypeError("path.paths: value must be a list of list[attribute].")
    for path_item in value:
      if not isinstance(path_item, list) or any(not isinstance(v, attribute) for v in path_item):
        raise TypeError("path.paths: each path must be List[attribute].")
    self.__paths = value

  @property
  def global_targets(self) -> List[attribute]:
    """Target attributes used for the full path discovery."""
    return self.__global_targets

  @property
  def local_targets(self) -> List[attribute]:
    """Subset of target attributes used for constrained differentiation."""
    return self.__local_targets

  @property
  def wrt_start_indices(self) -> List[int]:
    """Prefix offsets for gradient segments for each wrt attribute."""
    return self.__wrt_start_indices

  @property
  def wrt(self) -> List[attribute]:
    """Current differentiation variables represented by this path object."""
    return self.__wrt

  #########################################################
  ## Function to get the roots of an attribute
  ## as well as the path to that root
  #########################################################
  def __getRoots(self, att: attribute, parent_path: List[attribute], fixed_targets: List[attribute] = []) -> Tuple[List[attribute], List[List[attribute]]]:
    from yasps.attribute import JOIN, SUM, AVERAGE, DATA, UNION, CONSTANT
    stack: List[attribute] = [att]
    seenRoots: Set[attribute] = set([])
    roots: List[attribute] = []
    # we perform dfs to extract a path and its children
    while stack:
      current: attribute = stack.pop()
      if current in fixed_targets:
        # An explicitly requested target is a path leaf whether it is DATA or
        # CONSTANT.
        if current not in seenRoots:
          roots.append(current)
          seenRoots.add(current)
      elif len(fixed_targets) == 0 and current.operator == DATA:
        ## we got to the bottom of this path
        if current not in seenRoots:
          roots.append(current)
          seenRoots.add(current)
      elif current.operator == JOIN or current.operator == SUM or current.operator == AVERAGE:
        if current.through.dimension == 0:
          raise ValueError("path.__getRoots: att.through.dimension is 0. Such operation is not supported.")
        if current not in seenRoots:
          roots.append(current)
          seenRoots.add(current)
      elif current.operator == UNION:
        # we add the union operator to roots
        roots.append(current)
        seenRoots.add(current)
      else:
        stack.extend(current.children)
    # now we have the roots, which contains some joining operation
    # we need to duplicate the roots for those joining operation
    trueRoots: List[attribute] = []
    allPaths: List[List[attribute]] = []
    for root in roots:
      if root.operator == JOIN:
        # this is a joining operation
        # ok so at join, we will need to check how the joined attribute
        # will lead us to the final wrt attribute
        # TODO:
        # when the joined attribute is actually a union attribute
        # we will need to separate the paths into different possibilities
        # come back later for this
        childrenRoots, childrenPaths = self.__getRoots(root.children[0], [], fixed_targets)
        trueRoots += childrenRoots * root.through.dimension
        for childrenPath in childrenPaths:
          allPaths.append(parent_path + [root] + childrenPath)
        # allPaths += childrenPaths * root.through.dimension
      elif root.operator == UNION:
        # at union operator, we will need to add all the possible children paths
        for child in root.children:
          if child not in self.__unioned_child_to_its_children:
            self.__unioned_child_to_its_children[child] = []
          childrenRoots, childrenPaths = self.__getRoots(child, [], fixed_targets)
          trueRoots += childrenRoots
          for childrenPath in childrenPaths:
            allPaths.append(parent_path + [root] + childrenPath)
            if childrenPath[0] not in self.__unioned_child_to_its_children[child]:
              self.__unioned_child_to_its_children[child].append(childrenPath[0])
      elif root.operator == DATA or root.operator == CONSTANT:
        trueRoots.append(root)
        allPaths.append(parent_path + [root])
      else:
        raise ValueError(f"energy.getRoots: operator {root.operator} is not supported.")
    return trueRoots, allPaths


  def getRoots(self, att: attribute, parent_path: List[attribute], fixed_targets: List[attribute] = []):
    self.__root = att
    _, self.__paths = self.__getRoots(att, parent_path, fixed_targets)
    return

  #########################################################
  ## Path dict is used for differentiation
  ## it records for each boundary node, the children
  #########################################################
  def __get_path_dict(self):
    usedPaths: List[List[attribute]] = []
    local_targets_hashes = [x.hash for x in self.__local_targets]
    # we now always differentiate wrt all the data attributes
    # note that this excludes the constant attributes
    # after differentiation, we will decide which part of the matrix to put back in
    # if we simply cut off an attribute here, it will not be the full hessian we are projecting
    # as the eigen value we get will just be wrong
    for path in self.__paths:
      if path[-1].operator == DATA or path[-1].operator == CONSTANT:
        if len(self.__local_targets) != 0:
          if path[-1].hash in local_targets_hashes:
            usedPaths.append(path)
        else:
          usedPaths.append(path)

    # construct the path dict
    pathDict: Dict[attribute, List[attribute]] = {}
    # we convert the used path as a dictionary
    # by having the parent children relationship
    for path in usedPaths:
      if len(path) == 1:
        raise ValueError("energy.getSparseIndices: minimizing a data attribute as energy is not allowed.")
      for i in range(len(path) - 1):
        parent: attribute = path[i]
        child: attribute = path[i + 1]
        if parent not in pathDict:
          pathDict[parent] = []
        if child not in pathDict[parent]:
          pathDict[parent].append(child)
    self.__path_dict = pathDict
    return

  def getPathDict(self):
    self.__get_path_dict()

  def __jacobianLayoutRole(
    self,
    target: attribute,
    row_target_hashes: Set[int],
    column_target_hashes: Set[int]
  ) -> str:
    in_row = target.hash in row_target_hashes
    in_column = target.hash in column_target_hashes
    if in_row and in_column:
      return "rc"
    if in_row:
      return "r"
    if in_column:
      return "c"
    raise ValueError(
      "path.generateJacobianLayout: path contains terminal target "
      f"{target.fullName}, which is in neither row_targets nor column_targets."
    )

  def __jacobianLayoutRoleMask(self, role: str) -> int:
    if role == "r":
      return 1
    if role == "c":
      return 2
    if role == "rc":
      return 3
    raise ValueError(
      f"path.generateJacobianLayout: unsupported layout role {role}."
    )

  def __jacobianLayoutRoleFromMask(self, role_mask: int) -> str:
    if role_mask == 1:
      return "r"
    if role_mask == 2:
      return "c"
    if role_mask == 3:
      return "rc"
    raise ValueError(
      "path.generateJacobianLayout: a layout block must belong to at least "
      "one differentiation target set."
    )

  def __generateJacobianLayoutForAttribute(
    self,
    current: attribute,
    row_target_hashes: Set[int],
    column_target_hashes: Set[int]
  ) -> List[Tuple[str, int]]:
    if current.operator == DATA or current.operator == CONSTANT:
      return [(
        self.__jacobianLayoutRole(
          current,
          row_target_hashes,
          column_target_hashes
        ),
        current.size
      )]

    if current not in self.__path_dict:
      raise ValueError(
        "path.generateJacobianLayout: attribute "
        f"{current.fullName} is not present in path_dict."
      )

    if current.operator == UNION:
      # Differentiation pads every primitive branch of a union to the largest
      # branch Jacobian/Hessian span.  The union is consequently one block in
      # its parent's layout.  Its role is the union of all target roles that
      # can occupy that block.
      largest_branch_size = 0
      union_role_mask = 0
      children_on_path = self.__path_dict[current]
      for unioned_child in current.children:
        branch_layout: List[Tuple[str, int]] = []
        # get only this current unioned child's children
        branch_children = self.__unioned_child_to_its_children.get(
          unioned_child,
          []
        )
        for child in children_on_path:
          # we only care about this current unioned child's children
          if child in branch_children:
            branch_layout += self.__generateJacobianLayoutForAttribute(
              child,
              row_target_hashes,
              column_target_hashes
            )
        # get this current unioned child's branch's size
        branch_size = sum(item[1] for item in branch_layout)
        largest_branch_size = max(largest_branch_size, branch_size)
        for role, _ in branch_layout:
          union_role_mask |= self.__jacobianLayoutRoleMask(role)

      if largest_branch_size == 0:
        raise ValueError(
          "path.generateJacobianLayout: union attribute "
          f"{current.fullName} has no row or column target path."
        )
      return [(
        self.__jacobianLayoutRoleFromMask(union_role_mask),
        largest_branch_size
      )]

    children_layout: List[Tuple[str, int]] = []
    for child in self.__path_dict[current]:
      children_layout += self.__generateJacobianLayoutForAttribute(
        child,
        row_target_hashes,
        column_target_hashes
      )

    if current.operator == JOIN:
      return children_layout * current.through.dimension
    return children_layout

  def generateJacobianLayout(
    self,
    row_targets: List[attribute],
    column_targets: List[attribute]
  ) -> List[Tuple[str, int]]:
    """Return the combined Hessian's ordered row/column target blocks.

    Each tuple is ``(role, size)``.  ``role`` is ``"r"`` for a row-only
    block, ``"c"`` for a column-only block, and ``"rc"`` for a block that
    can contain targets from both sets.  Sizes are scalar axis spans in the
    uncompressed combined Hessian.
    """
    if not isinstance(row_targets, list) or len(row_targets) == 0:
      raise ValueError(
        "path.generateJacobianLayout: row_targets must be a non-empty list."
      )
    if not isinstance(column_targets, list) or len(column_targets) == 0:
      raise ValueError(
        "path.generateJacobianLayout: column_targets must be a non-empty list."
      )
    if (
      len(self.__path_dict) == 0
      or len(self.__paths) == 0
      or self.__root is None
    ):
      raise RuntimeError(
        "path.generateJacobianLayout: call getRoots and getPathDict first."
      )

    row_target_hashes = {target.hash for target in row_targets}
    column_target_hashes = {target.hash for target in column_targets}
    return self.__generateJacobianLayoutForAttribute(
      self.__root,
      row_target_hashes,
      column_target_hashes
    )
