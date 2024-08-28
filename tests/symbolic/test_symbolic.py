from yasps import symbolic

x = symbolic('x')
y = symbolic('y')
z = symbolic(y)
print(-x + y + 1 - z)
