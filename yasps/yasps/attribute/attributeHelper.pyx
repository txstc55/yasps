# cython: language_level=3
from typing import List
import yasps.attribute as ya
import hashlib # for hashing

def hashAttribute(att: ya.attribute) -> int:
  base_string = ""
  if att.correspondance is not None:
    base_string = att.correspondance.fullName
  if att.operator == ya.ADD or att.operator == ya.BROADCAST_ADD:
    att._attribute__hash = sum([child.hash for child in att.children])
  elif att.operator == ya.MUL:
    att._attribute__hash = att.children[0].hash * att.children[1].hash
  elif att.operator == ya.SUB or att.operator == ya.BROADCAST_SUB:
    att._attribute__hash = att.children[0].hash - att.children[1].hash
  elif att.operator == ya.DIV:
    division_string:str = f"{base_string}_{att.children[0].hash}/{att.children[1].hash}"
    att._attribute__hash = int(hashlib.sha256(division_string.encode()).hexdigest(), 16)
  elif att.operator == ya.POW:
    power_string:str = f"{base_string}_{att.children[0].hash}**{att.children[1].hash}"
    att._attribute__hash = int(hashlib.sha256(power_string.encode()).hexdigest(), 16)
  elif att.operator == ya.ATAN2:
    atan2_string:str = f"{base_string}_atan2({att.children[0].hash},{att.children[1].hash})"
    att._attribute__hash = int(hashlib.sha256(atan2_string.encode()).hexdigest(), 16)
  elif att.operator == ya.NEG:
    att._attribute__hash = -att.children[0].hash
  elif att.operator == ya.SIN:
    sin_string:str = f"{base_string}_sin({att.children[0].hash})"
    att._attribute__hash = int(hashlib.sha256(sin_string.encode()).hexdigest(), 16)
  elif att.operator == ya.COS:
    cos_string:str = f"{base_string}_cos({att.children[0].hash})"
    att._attribute__hash = int(hashlib.sha256(cos_string.encode()).hexdigest(), 16)
  elif att.operator == ya.TAN:
    tan_string:str = f"{base_string}_tan({att.children[0].hash})"
    att._attribute__hash = int(hashlib.sha256(tan_string.encode()).hexdigest(), 16)
  elif att.operator == ya.COT:
    cot_string:str = f"{base_string}_cot({att.children[0].hash})"
    att._attribute__hash = int(hashlib.sha256(cot_string.encode()).hexdigest(), 16)
  elif att.operator == ya.ABS:
    abs_string:str = f"{base_string}_abs({att.children[0].hash})"
    att._attribute__hash = int(hashlib.sha256(abs_string.encode()).hexdigest(), 16)
  elif att.operator == ya.LOG:
    log_string = f"{base_string}_ln({att.children[0].hash})"
    att._attribute__hash = int(hashlib.sha256(log_string.encode()).hexdigest(), 16)
  elif att.operator == ya.SELECT:
    select_string:str = f"{base_string}_select({att.children[0].hash},{att.children[1].hash},{att.children[2].hash})"
    att._attribute__hash = int(hashlib.sha256(select_string.encode()).hexdigest(), 16)
  elif att.operator == ya.SQRT:
    sqrt_string:str = f"{base_string}_sqrt({att.children[0].hash})"
    att._attribute__hash = int(hashlib.sha256(sqrt_string.encode()).hexdigest(), 16)
  elif att.operator == ya.EQ:
    eq_string:str = f"{base_string}_{att.children[0].hash} == ya.{att.children[1].hash}"
    att._attribute__hash = int(hashlib.sha256(eq_string.encode()).hexdigest(), 16)
  elif att.operator == ya.NEQ:
    ne_string:str = f"{base_string}_{att.children[0].hash} != {att.children[1].hash}"
    att._attribute__hash = int(hashlib.sha256(ne_string.encode()).hexdigest(), 16)
  elif att.operator == ya.GT:
    gt_string:str = f"{base_string}_{att.children[0].hash} > {att.children[1].hash}"
    att._attribute__hash = int(hashlib.sha256(gt_string.encode()).hexdigest(), 16)
  elif att.operator == ya.GEQ:
    ge_string:str = f"{base_string}_{att.children[0].hash} >= {att.children[1].hash}"
    att._attribute__hash = int(hashlib.sha256(ge_string.encode()).hexdigest(), 16)
  elif att.operator == ya.LT:
    lt_string:str = f"{base_string}_{att.children[0].hash} < {att.children[1].hash}"
    att._attribute__hash = int(hashlib.sha256(lt_string.encode()).hexdigest(), 16)
  elif att.operator == ya.LEQ:
    le_string:str = f"{base_string}_{att.children[0].hash} <= {att.children[1].hash}"
    att._attribute__hash = int(hashlib.sha256(le_string.encode()).hexdigest(), 16)
  elif att.operator == ya.ASSIGN:
    assign_string:str = f"{base_string}_{att.children[0].hash} = {att.children[1].hash}"
    att._attribute__hash = int(hashlib.sha256(assign_string.encode()).hexdigest(), 16)
  elif att.operator == ya.INDEX:
    return att.index_value
  elif att.operator == ya.ARRAY_ACCESS:
    array_access_string:str = f"{base_string}_{att.children[0].hash}[{att.children[1].hash}]"
    att._attribute__hash = int(hashlib.sha256(array_access_string.encode()).hexdigest(), 16)
  elif att.operator == ya.ARRAY:
    array_string:str = f"{base_string}_[{','.join([str(child.hash) for child in att.children])}]"
    att._attribute__hash = int(hashlib.sha256(array_string.encode()).hexdigest(), 16)
  elif att.operator == ya.FLOAT:
    float_str = format(att.float_value, '.17g').encode()
    # Compute the SHA-256 hash
    hash_hex = hashlib.sha256(float_str).hexdigest()
    # Convert the hexadecimal hash to an integer
    hash_int = int(hash_hex, 16)
    att._attribute__hash = hash_int
  elif att.operator == ya.DATA:
    fullname = str(att)
    # Compute the SHA-256 hash and convert to an integer
    hash_hex = hashlib.sha256(fullname.encode()).hexdigest()
    hash_int = int(hash_hex, 16)
    att._attribute__hash = hash_int
  elif att.operator == ya.JOIN or att.operator == ya.SUM or att.operator == ya.AVERAGE:
    if att.through is None:
      raise ValueError(f"attribute.hash: {att.operator.name.upper()} operator must have a through attribute.")
    operation_string:str = f"{base_string}_{att.operator.name}({att.children[0].hash}_through_{att.through.fullName})"
    att._attribute__hash = int(hashlib.sha256(operation_string.encode()).hexdigest(), 16)
  elif att.operator == ya.TRANSPOSE:
    transpose_string:str = f"{base_string}_transpose({att.children[0].hash})"
    att._attribute__hash = int(hashlib.sha256(transpose_string.encode()).hexdigest(), 16)
  elif att.operator == ya.ROW:
    row_string:str = f"{base_string}_{att.children[0].hash}.row({att.children[1].hash})"
    att._attribute__hash = int(hashlib.sha256(row_string.encode()).hexdigest(), 16)
  elif att.operator == ya.COL:
    col_string:str = f"{base_string}_{att.children[0].hash}.col({att.children[1].hash})"
    att._attribute__hash = int(hashlib.sha256(col_string.encode()).hexdigest(), 16)
  elif att.operator == ya.CROSS:
    cross_string:str = f"{base_string}_{att.children[0].hash}.cross({att.children[1].hash})"
    att._attribute__hash = int(hashlib.sha256(cross_string.encode()).hexdigest(), 16)
  elif att.operator == ya.NORM:
    norm_string:str = f"{base_string}_{att.children[0].hash}.norm()"
    att._attribute__hash = int(hashlib.sha256(norm_string.encode()).hexdigest(), 16)
  elif att.operator == ya.DET:
    det_string:str = f"{base_string}_{att.children[0].hash}.det()"
    att._attribute__hash = int(hashlib.sha256(det_string.encode()).hexdigest(), 16)
  elif att.operator == ya.INV:
    inv_string:str = f"{base_string}_{att.children[0].hash}.inv()"
    att._attribute__hash = int(hashlib.sha256(inv_string.encode()).hexdigest(), 16)
  elif att.operator == ya.DOT:
    dot_string:str = f"{base_string}_{att.children[0].hash}.dot({att.children[1].hash})"
    att._attribute__hash = int(hashlib.sha256(dot_string.encode()).hexdigest(), 16)
  elif att.operator == ya.RESIZE:
    resize_string: str = f"{base_string}_{att.children[0].hash}.resize({att.children[1].hash}, {att.children[2].hash})"
    att._attribute__hash = int(hashlib.sha256(resize_string.encode()).hexdigest(), 16)
  elif att.operator == ya.SPD:
    spd_string: str = f"{base_string}_{att.children[0].hash}.spd({att.children[1].hash})"
    att._attribute__hash = int(hashlib.sha256(spd_string.encode()).hexdigest(), 16)
  elif att.operator == ya.UNION:
    union_string: str = f"{base_string}_union({', '.join([str(x.hash) for x in att.children])})"
    att._attribute__hash = int(hashlib.sha256(union_string.encode()).hexdigest(), 16)
  return att._attribute__hash


def attribute2str(att: ya.attribute):
  if att.operator.type == 0:
    return f"{att.operator.name}({att.children[0]})"
  elif att.operator.type == 1:
    return f"({att.children[0]} {att.operator.name} {att.children[1]})"
  elif att.operator.type == 2:
    return f"{att.operator.name}({', '.join([str(child) for child in att.children])})"
  elif att.operator.type == 3:
    if att.operator == ya.INDEX:
      return str(att.index_value)
    elif att.operator == ya.FLOAT:
      return str(att.float_value)
    elif att.operator == ya.ARRAY_ACCESS:
      array_index = att.children[1].index_value
      row = array_index // att.children[0].cols
      col = array_index % att.children[0].cols
      return f"{att.children[0]}[{row}, {col}]"
    elif att.operator == ya.DATA:
      if att.correspondance is not None:
        return f"{att.correspondance.fullName}.{att.name}"
      else:
        raise ValueError("attribute.__str__: correspondance is None for a DATA attribute.")
    elif att.operator == ya.ARRAY:
      results = []
      for i in range(att.rows):
        row_strings: List[str] = []
        for j in range(att.cols):
          row_strings.append(str(att[i, j]))
        results.append(", ".join(row_strings))
      result_string = '\n'.join(results)
      return f"Mat(\n{result_string}\n)"
    elif att.operator == ya.JOIN or att.operator == ya.SUM or att.operator == ya.AVERAGE:
      if len(att.children) != 1:
        raise ValueError(f"attribute.__str__: {att.operator.name.upper()} operator must have one child.")
      if att.children[0].correspondance is None:
        raise ValueError(f"attribute.__str__: {att.operator.name.upper()} operator's first child must have a correspondance.")
      if att.through is None:
        raise ValueError(f"attribute.__str__: {att.operator.name.upper()} operator must have a through attribute.")
      return f"{att.operator.name}({att.children[0].fullName}->{att.correspondance.fullName}.{att.name})"
    elif att.operator == ya.ROW:
      if len(att.children) != 2:
        raise ValueError("attribute.__str__: ROW operator must have two children.")
      return f"{att.children[0]}.row({att.children[1]})"
    elif att.operator == ya.COL:
      if len(att.children) != 2:
        raise ValueError("attribute.__str__: COL operator must have two children.")
      return f"{att.children[0]}.col({att.children[1]})"
    elif att.operator == ya.TRANSPOSE:
      if len(att.children) != 1:
        raise ValueError("attribute.__str__: TRANSPOSE operator must have one child.")
      return f"{att.children[0]}.transpose()"
    elif att.operator == ya.BROADCAST_ADD:
      return f"{att.children[0]} + {att.children[1]}"
    elif att.operator == ya.BROADCAST_SUB:
      return f"{att.children[0]} - {att.children[1]}"
    elif att.operator == ya.CROSS:
      return f"{att.children[0]} x {att.children[1]}"
    elif att.operator == ya.NORM:
      return f"norm({att.children[0]})"
    elif att.operator == ya.DET:
      return f"det({att.children[0]})"
    elif att.operator == ya.INV:
      return f"inv({att.children[0]})"
    elif att.operator == ya.DOT:
      return f"{att.children[0]} · {att.children[1]}"
    elif att.operator == ya.RESIZE:
      return f"{att.children[0]}.resize({att.children[1]}, {att.children[2]})"
    elif att.operator == ya.SPD:
      return f"{att.children[0]}.spd({att.children[1]})"
    elif att.operator == ya.NEG:
      return f"-{att.children[0]}"
    elif att.operator == ya.UNION:
      return f"union({' || '.join([str(x) for x in att.children])})"
    else:
      raise ValueError(f"attribute.__str__: unknown operator type. Name: {att.operator.name}, Type: {att.operator.type}.")
  else:
    raise ValueError(f"attribute.__str__: unknown operator type. Name: {att.operator.name}, Type: {att.operator.type}.")


def checkHeritage(a1: ya.attribute, a2: ya.attribute) -> ya.attribute:
  # we check if two attribute are from the same line of blood
  # return the younger one always
  if a1.operator == ya.TRANSPOSE:
    return checkHeritage(a1.children[0], a2)
  if a2.operator == ya.TRANSPOSE:
    return checkHeritage(a1, a2.children[0])
  if a1.operator == ya.RESIZE:
    return checkHeritage(a1.children[0], a2)
  if a2.operator == ya.RESIZE:
    return checkHeritage(a1, a2.children[0])
  # add special case for select
  if a1.operator == ya.SELECT and a1.correspondance is None:
    return a2
  if a2.operator == ya.SELECT and a2.correspondance is None:
    return a1
  if a1.correspondance is None and (a1.operator != ya.FLOAT and not a1.isFloatMat):
    raise ValueError(f"attribute.__check_heritage: a1 must have a correspondance since it is not a float value. a1 is: {a1}")
  if a2.correspondance is None and (a2.operator != ya.FLOAT and not a2.isFloatMat):
    raise ValueError(f"attribute.__check_heritage: a2 must have a correspondance since it is not a float value. a2 is: {a2}")
  if a1.operator == ya.FLOAT:
    return a2
  if a2.operator == ya.FLOAT:
    return a1

  if a1.isFloatMat:
    return a2
  if a2.isFloatMat:
    return a1

  if a1.correspondance is None or a2.correspondance is None:
    raise ValueError("attribute.__check_heritage: correspondance should be set for both attributes.")
  else:
    if a1.correspondance.fullName == a2.correspondance.fullName:
      # same correspondance, we can return either one
      return a1
    if a1.correspondance.type == "scene":
      # a1 is a scene, we check if a2 is a child of a1
      if a2.correspondance.scene.fullName == a1.correspondance.fullName:
        return a2
    if a2.correspondance.type == "scene":
      # print("Enter here?")
      # same scenario
      if a1.correspondance.scene.fullName == a2.correspondance.fullName:
        return a1
    # now we actually need to check the heritage
    if a1.correspondance.type == "mesh":
      if a2.correspondance.mesh.fullName == a1.correspondance.fullName:
        return a2
    if a2.correspondance.type == "mesh":
      if a1.correspondance.mesh.fullName == a2.correspondance.fullName:
        return a1
    # we dont need to check for primitives, sicen if they are the same
    # then we already checked it
    # if they are not the same, we raise error anyway
    raise ValueError("attribute.__check_heritage: attributes do not share the same heritage.")
