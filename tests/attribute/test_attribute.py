from yasps import scene
from yasps import codeGenerator
import numpy as np
# set seed
np.random.seed(0)

scene0 = scene("scene0") # create the scene
m1 = scene0.addMesh("mesh1") # add the mesh
box_vertices = m1.addPrimitive("box_vertices", 8) # add a primitive box vertex as control points
box_vertices.addAttribute("position", rows = 1, cols = 3) # box vertices have 3D positions

# update the values
positions = scene0.mesh1.box_vertices["position"]
positions.updateValue(np.array([[-1, 1, 1], [1, 1, 1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, -1, -1], [-1, -1, -1]]))


# now create a d8 mesh
m1.addPrimitive("vertices", 6) # d8

# create the relationship between the vertex to box vertices
m1.vertices.addConnectivity("box_to_vertex", m1.box_vertices, np.array([[0, 1, 2, 3], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [0, 4, 7, 3], [4, 5, 6, 7]]), 4)

# now we define that position of d8 is correlated to box vertices
m1.vertices.addAttribute("position", through = m1.vertices.box_to_vertex)

# now we add the weights
# each weight is a 1 by 4 vector
# since each vertex corresponds to 4 box vertices
m1.vertices.addAttribute("box_vertex_weights", rows = 1, cols = 4)
m1.vertices["box_vertex_weights"].updateValue(np.array([[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]))

# here we compute the weighted position
weighted_position = m1.vertices["position"].row(0) * m1.vertices["box_vertex_weights"][0] + m1.vertices["position"].row(1) * m1.vertices["box_vertex_weights"][1] + m1.vertices["position"].row(2) * m1.vertices["box_vertex_weights"][2] + m1.vertices["position"].row(3) * m1.vertices["box_vertex_weights"][3]

m1.vertices.addAttribute("weighted_position", computed_attribute = weighted_position)

# print(weighted_position)


attr = scene0.mesh1.box_vertices["position"] * 3.0
# attr = scene0.mesh1.vertices["position"].row(0)
# attr = m1.vertices["weighted_position"]
attr.compute()
