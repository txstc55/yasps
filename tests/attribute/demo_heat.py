from yasps import scene
from yasps import codeGenerator
import numpy as np
import time

# first we import bunny obj
f = open('../data/bunny.obj', 'r')
# read the vertices and faces
vertices = []
faces = []
for line in f:
  if line[0] == 'v':
    vertices.append(list(map(float, line[2:].split())))
  if line[0] == 'f':
    faces.append(list(map(int, line[2:].split())))
f.close()

# make it 0 based
faces = np.array(faces) - 1

# create the vertex id to vertex id connectivity
vertexToVertex = []
for i in range(len(vertices)):
  vertexToVertex.append([])

for face in faces:
  for i in range(3):
    vertexToVertex[face[i]].append(face[(i + 1) % 3])
    vertexToVertex[face[i]].append(face[(i + 2) % 3])

# initialize the scene
s0 = scene("scene0")
s0.addAttribute("dt", rows = 1, cols = 1)
s0.attributes["dt"].updateValue(np.array([0.001]))

# add the bunny mesh
bunny = s0.addMesh("bunny")
bunny_vertices = bunny.addPrimitive("vertices", numInstances = len(vertices))

# add the vertex positions
vertex_positions = bunny_vertices.addAttribute("position", rows = 1, cols = 3)
vertex_positions.updateValue(vertices)
bunny_vertices.addConnectivity("vertex_to_vertex", bunny_vertices, vertexToVertex, 0)

# add the heat attribute
bunny_vertices.addAttribute("heat", rows = 1, cols = 1)
# heat_values = np.random.rand(len(vertices))
heat_values = np.zeros(len(vertices))
heat_values[0:len(vertices):2] = 1.0
bunny_vertices.attributes["heat"].updateValue(heat_values)

# generate average around neighbor
avg_heat = bunny_vertices.addAttribute("avg_heat", through = bunny_vertices.connectivities["vertex_to_vertex"], operation = "AVERAGE", source = bunny_vertices["heat"])

# get the d_heat
heat_change = (avg_heat - bunny_vertices["heat"]) * s0["dt"]
bunny_vertices.addAttribute("d_heat", computed_attribute = heat_change)


import pyvista as pv
triangles = np.array(faces)
cells = np.hstack([np.full((triangles.shape[0], 1), 3), triangles])
heatmap = bunny_vertices.attributes["heat"].value.get()
mesh = pv.PolyData(np.array(vertices), cells)
mesh.point_data["heat"] = heatmap

plotter = pv.Plotter()
actor = plotter.add_mesh(mesh, scalars="heat")

heat_change.compute() # pre generate the code
plotter.show(interactive_update=True)


total_frames = 0
start = time.time()
def update_heat():
  global total_frames, start
  change_value = heat_change.compute().value
  heat = (bunny_vertices.attributes["heat"].value.get() + change_value.get())
  bunny_vertices.attributes["heat"].updateValue(heat)
  mesh.point_data["heat"] = heat
  plotter.update_scalars(heat)
  total_frames += 1
  if total_frames % 10 == 0:
    end = time.time()
    print(f"Time taken for 10 frames: {end - start}")
    start = time.time()

while True:
  update_heat()
