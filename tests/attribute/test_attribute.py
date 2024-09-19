from yasps import scene
import numpy as np
# set seed
np.random.seed(0)

scene0 = scene("scene0")
m1 = scene0.addMesh("mesh1")
m1.addAttribute("a1", rows = 3, cols = 3)
box_vertices = m1.addPrimitive("box_vertices", 8)
box_vertices.addAttribute("position", rows = 1, cols = 3)
positions = scene0.mesh1.box_vertices["position"]
positions.updateValue(np.array([[-1, 1, 1], [1, 1, 1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, -1, -1], [-1, -1, -1]]))


m1.addPrimitive("vertices", 5) # d6 somehow
m1.vertices.addConnectivity("box_to_vertex", m1.box_vertices, np.array([[0, 1, 2, 3], [0, 4, 7, 3], [1, 5, 6, 2], [2, 6, 7, 3], [4, 5, 6, 7]]), 4)
m1.vertices.addAttribute("box_vertex_weights", rows = 1, cols = 4)

# generate a 5 by 4 matrix of random numbers
weights = np.random.rand(5, 4)
# normalize the weight at each vertex
weights = weights / np.sum(weights, axis = 1)[:, np.newaxis]
# print(weights)
m1.vertices["box_vertex_weights"].updateValue(weights)
m1.vertices.addAttribute("position", through = m1.vertices.box_to_vertex)
print(m1.vertices["position"].row(0)[0])



# print(scene0.mesh1.box_vertices["position"].value.get())
