from yasps.scene import scene
from yasps.autodiff import autodiff


s = scene("test")
a = s.addAttribute("a", rows = 2, cols = 2)
b = s.addAttribute("b", rows = 2, cols = 2)

# print(a.mul_explicit(b))

ad = autodiff()
result = ad.diff(a * b, a)
print(result)
