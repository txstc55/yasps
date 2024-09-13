from yasps import scene


scene0 = scene("scene0")
m1 = scene0.addMesh("mesh1")
m1.addAttribute("a1", rows = 3, cols = 3)
m1v = m1.addPrimitive("vertex", 10)
m1v.addAttribute("position", rows = 1, cols = 3)

positions = scene0.mesh1.vertex["position"]
positions.updateValue([1.0, 2.0, 3.0])
print(scene0.mesh1.vertex["position"].value.get())
