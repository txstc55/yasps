from .symbolic import symbolic

class symbolicMatrix:
  symbolicMatrixSize = {} # dictionary to store the size of the matrix
  def __init__(self, name: str, rows: int = 1, cols: int = 1, data = [], correspondance: list[str] = []):
    if data != None:
      # single element to a 1x1 matrix
      if (not isinstance(data, list)):
        self.__rows = 1
        self.__cols = 1
        self.__data = [[symbolic(data, correspondance = correspondance)]]
        symbolicMatrix.symbolicMatrixSize[name] = (1, 1)
      ## check if it is an 1d array
      elif (not isinstance(data[0], list)):
        self.__rows = 1
        self.__cols = len(data)
        self.__data = [[symbolic(data[i], correspondance = correspondance) for i in range(len(data))]]
        symbolicMatrix.symbolicMatrixSize[name] = (1, len(data))
      elif (isinstance(data[0], list)):
        self.__rows = len(data)
        self.__cols = len(data[0])
        self.__data = [[symbolic(data[j][i], correspondance = correspondance) for i in range(cols)] for j in range(rows)]
        symbolicMatrix.symbolicMatrixSize[name] = (rows, cols)
      else:
          raise ValueError("SymbolicMatrix: Invalid input data, supported are numbers, 1d arrays and 2d arrays")
      return

    self.__name = name
    self.__rows = rows
    self.__cols = cols
    self.__data = [[symbolic(name, operator = symbolic.array_access, children = [symbolic(j * rows + i)]) for i in range(cols)] for j in range(rows)]
    symbolicMatrix.symbolicMatrixSize[name] = (rows, cols)


  @property
  def name(self)->str:
    return self.__name

  @property
  def rows(self)->int:
    return self.__rows

  @property
  def cols(self)->int:
    return self.__cols

  @staticmethod
  def getMatrixSize(name):
    if name not in symbolicMatrix.symbolicMatrixSize:
      raise ValueError(f"symbolicMatrix.getMatrixSize: Matrix {name} not found")
    return symbolicMatrix.symbolicMatrixSize[name]

  @property
  def data(self):
    return self.__data
  def __getitem__(self, index):
    if isinstance(index, int):
      if (index >= self.rows * self.cols):
        raise ValueError(f"SymbolicMatrix.__getitem__: Index out of bounds, index = index, {self.name}.rows * cols = {self.rows * self.cols}")
      row = index // self.cols
      col = index % self.cols
      return self.data[row][col]
    if isinstance(index, tuple):
      row = index[0]
      col = index[1]
      if (row > self.rows):
        raise ValueError(f"SymbolicMatrix.__getitem__: Row index out of bounds, row = {row}, {self.name}.rows = {self.rows}")
      if (col > self.cols):
        raise ValueError(f"SymbolicMatrix.__getitem__: Column index out of bounds, col = {col}, {self.name}.cols = {self.cols}")
      return self.data[index[0]][index[1]]

  def __setitem__(self, index, value):
    if isinstance(index, int):
      if (index >= self.rows * self.cols):
        raise ValueError(f"SymbolicMatrix.__setitem__: Index out of bounds, index = index, {self.name}.rows * cols = {self.rows * self.cols}")
      row = index // self.cols
      col = index % self.cols
      self.data[row][col] = symbolic(value)
    if isinstance(index, tuple):
      row = index[0]
      col = index[1]
      if (row > self.rows):
        raise ValueError(f"SymbolicMatrix.__setitem__: Row index out of bounds, row = {row}, {self.name}.rows = {self.rows}")
      if (col > self.cols):
        raise ValueError(f"SymbolicMatrix.__setitem__: Column index out of bounds, col = {col}, {self.name}.cols = {self.cols}")
      self.data[row][col] = symbolic(value)
