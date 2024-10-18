import yasps.attribute as ya
from typing import Union
import math
def add(att: ya.attribute, other: ya.attribute) -> ya.attribute:
  if other.isZero:
    return att
  if att.isZero:
    return other
  if att.operator == ya.FLOAT and other.operator == ya.FLOAT:
    return ya.attribute(float_value = att.float_value + other.float_value)
  if att.size == 1 and other.size == 1:
    return ya.attribute(children = [att, other], operator = ya.ADD, correspondance = ya.attribute._attribute__check_heritage(att, other).correspondance, rows = 1, cols = 1)
  if other.size == 1:
    return ya.attribute(children = [att, other], operator = ya.BROADCAST_ADD, correspondance = ya.attribute._attribute__check_heritage(att, other).correspondance, rows = att.rows, cols = att.cols)
  if att.size == 1:
    return ya.attribute(children = [other, att], operator = ya.BROADCAST_ADD, correspondance = ya.attribute._attribute__check_heritage(att, other).correspondance, rows = other.rows, cols = other.cols)
  # check dimension match
  if att.rows == other.rows and att.cols == other.cols:
    return ya.attribute(children = [att, other], operator = ya.ADD, correspondance = ya.attribute._attribute__check_heritage(att, other).correspondance, rows = att.rows, cols = att.cols)
  else:
    raise ValueError("attribute.__add__: cannot add two attributes of different dimensions.")

def add_explicitly(att: ya.attribute, other: ya.attribute) -> ya.attribute:
  if other.isZero:
    return att
  if att.isZero:
    return other
  if att.size == 1 and other.size == 1:
    return att + other
  if other.size == 1:
    return ya.attribute.to_array([att[i] + other for i in range(att.size)], att.rows, att.cols)
  if att.size == 1:
    return ya.attribute.to_array([att + other[i] for i in range(other.size)], other.rows, other.cols)
  if att.rows == other.rows and att.cols == other.cols:
    return ya.attribute.to_array([att[i] + other[i] for i in range(att.size)], att.rows, att.cols)
  else:
    raise ValueError("attribute.add_explicit: cannot add two attributes of different dimensions.")

def sub(att: ya.attribute, other: ya.attribute) -> ya.attribute:
  if other.isZero:
    return att
  if att.isZero:
    return -other
  if att.hash == other.hash:
    return ya.attribute.zeros(att.rows, att.cols)
  if att.operator == ya.FLOAT and other.operator == ya.FLOAT:
    return ya.attribute(float_value = att.float_value - other.float_value)
  if att.size == 1 and other.size == 1:
    return ya.attribute(children = [att, other], operator = ya.SUB, correspondance = ya.attribute._attribute__check_heritage(att, other).correspondance, rows = 1, cols = 1)
  if other.size == 1:
    return ya.attribute(children = [att, other], operator = ya.BROADCAST_SUB, correspondance = ya.attribute._attribute__check_heritage(att, other).correspondance, rows = att.rows, cols = att.cols)
  if att.size == 1:
    return ya.attribute(children = [-att, other], operator = ya.BROADCAST_ADD, correspondance = ya.attribute._attribute__check_heritage(att, other).correspondance, rows = other.rows, cols = other.cols)
  # check dimension match
  if att.rows == other.rows and att.cols == other.cols:
    return ya.attribute(children = [att, other], operator = ya.SUB, correspondance = ya.attribute._attribute__check_heritage(att, other).correspondance, rows = att.rows, cols = att.cols)
  else:
    raise ValueError("attribute.__sub__: cannot subtract two attributes of different dimensions.")

def sub_explicitly(att: ya.attribute, other: ya.attribute) -> ya.attribute:
  if other.isZero:
    return att
  if att.isZero:
    return -other
  if att.size == 1 and other.size == 1:
    return att - other
  if other.size == 1:
    return ya.attribute.to_array([att[i] - other for i in range(att.size)], att.rows, att.cols)
  if att.size == 1:
    return ya.attribute.to_array([att - other[i] for i in range(other.size)], other.rows, other.cols)
  if att.rows == other.rows and att.cols == other.cols:
    return ya.attribute.to_array([att[i] - other[i] for i in range(att.size)], att.rows, att.cols)
  else:
    raise ValueError("attribute.sub_explicit: cannot subtract two attributes of different dimensions.")

def mul(att: ya.attribute, other: ya.attribute) -> ya.attribute:
  if att.isIdentity:
    return other
  if other.isIdentity:
    return att
  if att.isZero:
    return ya.attribute.zeros(other.rows, other.cols)
  if other.isZero:
    return ya.attribute.zeros(att.rows, att.cols)
  if other.operator == ya.FLOAT and att.operator == ya.FLOAT:
    return ya.attribute(float_value = att.float_value * other.float_value)
  if att.size == 1 or other.size == 1:
    return ya.attribute(children = [att, other], operator = ya.MUL, correspondance = ya.attribute._attribute__check_heritage(att, other).correspondance, rows = max(att.rows, other.rows), cols = max(att.cols, other.cols))
  if att.cols == other.rows:
    return ya.attribute(children = [att, other], operator = ya.MUL, correspondance = ya.attribute._attribute__check_heritage(att, other).correspondance, rows = att.rows, cols = other.cols)
  else:
    raise ValueError(f"attribute.__mul__: dimension mismatch, cannot multiply {att.rows}x{att.cols} with {other.rows}x{other.cols}.")


def mul_explicitly(att: ya.attribute, other: ya.attribute) -> ya.attribute:
  if att.isIdentity:
    return other
  if other.isIdentity:
    return att
  if att.isZero:
    return ya.attribute.zeros(other.rows, other.cols)
  if other.isZero:
    return ya.attribute.zeros(att.rows, att.cols)

  if att.operator == ya.FLOAT and other.operator == ya.FLOAT:
    return ya.attribute(float_value = att.float_value * other.float_value)
  if att.size == 1:
    return ya.attribute.to_array([att * other[i] for i in range(other.size)], other.rows, other.cols)
  if other.size == 1:
    return ya.attribute.to_array([att[i] * other for i in range(att.size)], att.rows, att.cols)
  if att.cols == other.rows:
    # start the triple for loop
    result = ya.attribute.zeros(att.rows, other.cols)
    for i in range(att.rows):
      for j in range(other.cols):
        for k in range(att.cols):
          result.children[i * other.cols + j] += att[i * att.cols + k] * other[k * other.cols + j]
    return result
  else:
    raise ValueError(f"attribute.mul_explicit: dimension mismatch, cannot multiply {att.rows}x{att.cols} with {other.rows}x{other.cols}.")


def div(att: ya.attribute, other: ya.attribute):
  if other.size != 1:
    raise ValueError("attribute.__div__: cannot divide by a non scalar.")
  if other.isZero:
    raise ValueError("attribute.__div__: cannot divide by zero.")
  if other.isIdentity:
    return att
  if other.operator == ya.FLOAT:
    return att * (1.0 / other.float_value)
  return att * (1.0 / other)

def div_explicitly(att: ya.attribute, other: ya.attribute):
  if other.size != 1:
    raise ValueError("attribute.div_explicit: cannot divide by a non scalar.")
  if other.isZero:
    raise ValueError("attribute.div_explicit: cannot divide by zero.")
  if other.isIdentity:
    return att
  if other.operator == ya.FLOAT:
    return mul_explicitly(att, ya.attribute(float_value = 1.0 / other.float_value))
  return ya.attribute.to_array([att[i] * (1.0 / other) for i in range(att.size)], att.rows, att.cols)

def pow_op(att: ya.attribute, other: ya.attribute):
  if att.size != 1:
    raise ValueError("attribute.__pow__: cannot raise power of a non scalar.")
  if other.size != 1:
    raise ValueError("attribute.__pow__: cannot raise to a non scalar power.")
  if att.isZero:
    return ya.attribute(float_value = 0.0)
  if att.isIdentity:
    return ya.attribute(float_value = 1.0)
  if other.isZero:
    return ya.attribute(float_value = 1.0)
  if other.isIdentity:
    return att
  if att.operator == ya.FLOAT and other.operator == ya.FLOAT:
    return ya.attribute(float_value = att.float_value ** other.float_value)
  return ya.attribute(children = [att, other], operator = ya.POW, correspondance = ya.attribute._attribute__check_heritage(att, other).correspondance, rows = att.rows, cols = att.cols)

def sqrt_op(att: ya.attribute):
  if att.size != 1:
    raise ValueError("attribute.__sqrt__: cannot take square root of a non scalar.")
  if att.isZero:
    return ya.attribute(float_value = 0.0)
  if att.isIdentity:
    return ya.attribute(float_value = 1.0)
  if att.operator == ya.FLOAT:
    return ya.attribute(float_value = math.sqrt(att.float_value))
  return ya.attribute(children = [att], operator = ya.SQRT, correspondance = att.correspondance, rows = att.rows, cols = att.cols)

def log_op(att: ya.attribute):
  if att.size != 1:
    raise ValueError("attribute.__log__: cannot take log of a non scalar.")
  if att.isZero:
    raise ValueError("attribute.__log__: cannot take log of zero.")
  if att.isIdentity:
    return ya.attribute(float_value = 0.0)
  if att.operator == ya.FLOAT:
    return ya.attribute(float_value = math.log(att.float_value))
  return ya.attribute(children = [att], operator = ya.LOG, correspondance = att.correspondance, rows = att.rows, cols = att.cols)
