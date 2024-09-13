from yasps import scene
# from yasps import mesh


scene0 = scene("scene0")
m1 = scene0.addMesh("mesh1")
m1.addAttribute("a1", rows = 3, cols = 3)
m1v = m1.addPrimitive("vertex", 10)
m1v.addAttribute("position", rows = 1, cols = 3)


print(scene0.mesh1.vertex["position"])
