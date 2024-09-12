# cython: language_level=3
# operator is used to define what property
# an operator should have
# and how the string should be generated
# from the operator and its children
class operator:
  def __init__(self, name: str, operator_type: int, commutative: bool):
    self.__name = name
    self.__operator_type = operator_type
    self.__commutative = commutative

  @property
  def name(self)->str:
    return self.__name

  @property
  def type(self)->int:
    # 0 for op(x)
    # 1 for x op y
    # 2 for op($0, $1, ...)
    # 3 for other things
    return self.__operator_type


  @property
  def commutative(self)->bool:
    return self.__commutative


  def __str__(self)->str:
    return f"operator({self.__name}, type={self.type}, commutative={self.commutative})"

  # check if two operators are equal
  def __eq__(self, other)->bool:
    if self.name == other.name and self.type == other.type and self.commutative == other.commutative:
      return True
    return False
