# cython: language_level=3

class operator:
  def __init__(self, operator_symbol: str, operator_type: int, commutative: bool):
    self.__operator_symbol = operator_symbol
    self.__operator_type = operator_type
    self.__commutative = commutative

  @property
  def symbol(self)->str:
    return self.__operator_symbol

  @property
  def type(self):
    # 0 for op(x)
    # 1 for x op y
    # 2 for op($0, $1, ...)
    # 3 for naming
    return self.__operator_type


  @property
  def commutative(self):
    return self.__commutative


  def to_string(self, children)->str:
    if self.__commutative:
      return self.symbol.join([child.to_string() for child in children])
    else:
      if self.type == 0:
        return f"{self.symbol}({children[0].to_string()})"
      elif self.type == 1:
        return f"(({children[0].to_string()}) {self.symbol} ({children[1].to_string()}))"
      elif self.type == 2:
        return f"{self.symbol}({', '.join([child.to_string() for child in children])})"
      elif self.type == 3:
        # this is used for naming
        # so the children is either a string, or a number maybe
        # or it can be an intermediate
        return children[0].to_string()
      else:
        raise ValueError(f"operator.to_string: Invalid operator type {self.symbol}")
