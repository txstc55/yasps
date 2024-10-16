from yasps.scene import scene
from yasps.autodiff import autodiff


s = scene("test")
a = s.addAttribute("a", rows = 2, cols = 2)
b = s.addAttribute("b", rows = 2, cols = 2)
ad = autodiff()

## test multiplication
result = ad.diff(a * b, a)
print(result)

# test division
a2 = a * a
a0 = a2[0, 0]
result = ad.diff(a / a[0, 0], a)
print(result)
# test =  -1 * (1.0 / (a[0, 0] * a[0, 0]))
# print(test)
