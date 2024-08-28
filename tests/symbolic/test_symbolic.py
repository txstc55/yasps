from yasps import symbolic

x = symbolic('x')
y = symbolic('y')
z = -symbolic(y)
print(-x + y + 1 - symbolic.sin(z))
print(symbolic.select(x, y, z))

result: symbolic = symbolic(1)
for i in range(100):
  if i % 2 == 0:
    result *= x
  else:
    result *= y

print(result.children, result.operator)
