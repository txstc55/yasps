# Attribute API

Attributes are normally constructed through `scene`, `mesh`, `primitive`, or
`primitiveUnion`. Import the class for static constructors:

```python
from yasps.attribute import attribute
```

## Data and evaluation

| Member | Description |
| --- | --- |
| `updateValue(value, deepCopy=False)` | Replace/alias numerical data |
| `compute()` | Evaluate and return this attribute |
| `value` | Backend device-array result |
| `rows`, `cols`, `size` | Per-instance shape |
| `correspondance` | Owning scene/mesh/primitive |
| `name`, `fullName` | User and generated identity |
| `operator`, `children` | Symbolic graph inspection |

## Construction and shape

```python
attribute.to_array(children, rows, cols)
attribute.zeros(rows, cols)
attribute.identity(rows)
expression.resize(rows, cols)
```

`resize()` returns a new symbolic expression. `reshape()` mutates metadata and
returns `None`.

## Arithmetic

```python
a + b
a - b
-a
a * b
a / scalar
a.pow(exponent)
```

Non-scalar `*` is matrix multiplication.

The `add_explicit`, `sub_explicit`, `mul_explicit`, and `div_explicit`
variants eagerly construct expanded symbolic expressions. They are mainly
used by differentiation internals; `mul_explicit` is still matrix/scalar
algebra, not Hadamard multiplication.

## Scalar functions

```python
x.sqrt()
x.log()
x.sin()
x.cos()
y.atan2(x)
x.abs()
```

Arguments must have supported scalar/vector shapes.

## Matrix and vector functions

```python
matrix.transpose()
matrix.trace()
matrix.inverse()
matrix.determinant()
matrix.spd(method)

vector.norm()
left.dot(right)
left.cross(right)

matrix.row(index)
matrix.col(index)
matrix[row, column]
```

## Conditions

```python
left.eq(right)
left.neq(right)
left > right
left >= right

attribute.select(condition, true_value, false_value)
```

Do not use Python `==` to create a numerical expression.

## Derivative control

```python
frozen = expression.asConstant()
```

This blocks differentiation through `expression` without freezing its
eventual numerical evaluation.

## Advanced metadata

`generate_code`, `disable_array_access`, `setAsIntermediate()`, device/global
kernel properties, hashes, and generated data names are internal code
generation controls. Model code should not rely on them.
