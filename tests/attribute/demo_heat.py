from yasps import scene
from yasps import codeGenerator
import numpy as np

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

# add the bunny mesh
bunny = s0.addMesh("bunny")
bunny_vertices = bunny.addPrimitive("vertices", numInstances = len(vertices))
vertex_positions = bunny_vertices.addAttribute("position", rows = 1, cols = 3)
vertex_positions.updateValue(vertices)
bunny_vertices.addConnectivity("vertex_to_vertex", bunny_vertices, vertexToVertex, 0)

# add the heat attribute
bunny_vertices.addAttribute("heat", rows = 1, cols = 1)
