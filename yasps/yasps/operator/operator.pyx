# cython: language_level=3

cdef class operator:
  cdef readonly str _name
  cdef readonly int _operator_type
  cdef readonly bint _commutative

  def __init__(self, str name, int operator_type, bint commutative):
    self._name = name
    self._operator_type = operator_type
    self._commutative = commutative

  @property
  def name(self) -> str:
    return self._name

  @property
  def type(self) -> int:
    """0: op(x), 1: x op y, 2: op($0, $1, ...), 3: other"""
    return self._operator_type

  @property
  def commutative(self) -> bint:
    return self._commutative

  def __richcmp__(self, other, int op):
    if op == 2:  # == operator
      return self._is_equal(other)
    elif op == 3:  # != operator
      return not self._is_equal(other)
    else:
      return NotImplemented

  cdef bint _is_equal(self, operator other):
    return (self._name == other._name and
            self._operator_type == other._operator_type and
            self._commutative == other._commutative)

  def __str__(self) -> str:
    return f"operator({self.name}, type={self.type}, commutative={self.commutative})"
