# Operator support

This table describes the public symbolic surface, not merely operator
constants that exist internally.

| Operation | Public form | Compute | Scene differentiation | Notes |
| --- | --- | :---: | :---: | --- |
| Add/subtract | `a + b`, `a - b` | yes | yes | Scalar broadcast is shape-dependent |
| Negate | `-a` | yes | yes | |
| Multiply | `a * b` | yes | yes | Scalar scale or matrix product |
| Divide | `a / scalar` | yes | yes | Denominator scalar |
| Power | `a.pow(b)` | yes | yes | Scalar base/exponent |
| Square root | `a.sqrt()` | yes | yes | Scalar |
| Log | `a.log()` | yes | yes | Scalar |
| Sin/cos | `a.sin()`, `a.cos()` | yes | yes | Scalar |
| atan2 | `y.atan2(x)` | yes | yes | Scalars |
| Absolute | `a.abs()` | yes | yes | Derivative at zero uses the convention `0` |
| Dot | `a.dot(b)` | yes | yes | Equal-size vectors |
| Cross | `a.cross(b)` | yes | yes | Three components |
| Norm | `a.norm()` | yes | yes | Vector |
| Transpose | `a.transpose()` | yes | yes | |
| Row/column | `a.row(i)`, `a.col(i)` | yes | yes | Static index |
| Array access | `a[i]`, `a[i,j]` | yes | yes | Per-instance element |
| Trace | `a.trace()` | expanded | yes | Square matrix |
| Determinant | `a.determinant()` | yes | yes | Square matrix |
| Inverse | `a.inverse()` | yes | yes | Square matrix |
| PSD projection | `a.spd(method)` | yes | not as input op | Used on Hessians |
| Equality | `a.eq(b)`, `a.neq(b)` | yes | condition only | Python `==` is structural |
| Greater | `a > b`, `a >= b` | yes | condition only | Scalars/shapes must match |
| Select | `attribute.select(c,t,f)` | yes | branch derivatives | Same branch shapes |
| JOIN | `addAttribute(... through=...)` | yes | yes | Fixed arity |
| SUM/AVERAGE | `operation="..."` | yes | limited | Variable-arity derivatives unsupported |
| UNION | `primitiveUnion.addAttribute(name)` | yes | yes | Matching child shape/name |
| Stop gradient | `a.asConstant()` | yes | yes | Child derivative becomes zero |

`TAN`, `COT`, `<`, and `<=` operator objects exist internally, but the current
attribute class does not expose corresponding public methods/operators.

Elementwise matrix multiplication and an exponential function are not part of
the current public surface.
