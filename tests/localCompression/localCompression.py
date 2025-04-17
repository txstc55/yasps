from yasps import scene

s0 = scene("scene0")

m1 = s0.addMesh("m1")

v = m1.addPrimitive("vertices", numInstances = 5)
vp = v.addAttribute("position", rows = 3, cols = 1)
vs = v.addAttribute("size", rows = 1, cols = 1)
v["position"].updateValue([0.2, 1.3, 2.4] * 5)
v["size"].updateValue([1, 2, 2, 1, 4.5])

f = m1.addPrimitive("faces", numInstances = 6)
f2v = f.addConnectivity("face_to_vertex", v, [[0, 1, 2, 3], [0, 2, 0, 3], [2, 2, 1, 2], [4, 4, 4, 4], [2, 1, 3, 4], [2, 1, 2, 3]], 4) # 1 face connects to 4 vertices
# 4, 3, 2, 1, 4, 3

fp = f.addAttribute("position", through = f2v, source = v["position"])
fs = f.addAttribute("size", through = f2v, source = v["size"])

fa = ((fp.row(0).cross(fp.row(1)) + fp.row(1).cross(fp.row(2)) + fp.row(2).cross(fp.row(3)) + fp.row(3).cross(fp.row(0)))).norm() * (fs.row(0) + fs.row(1))

f.addAttribute("area", computed_attribute = fa)


s0.addEnergy(fa)
s0.addMinimizeTarget([vp, vs])
