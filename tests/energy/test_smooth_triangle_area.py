from yasps import scene
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
vertices = np.array(vertices) * 1000

# initialize the scene
s0 = scene("scene0")
# add the bunny mesh
bunny = s0.addMesh("bunny")
bunny_vertices = bunny.addPrimitive("vertices", numInstances = len(vertices))
bunny_vertices_positions = bunny_vertices.addAttribute("position", rows = 1, cols = 3)
bunny_vertices_positions.updateValue(vertices)
bunny_faces = bunny.addPrimitive("faces", numInstances = len(faces))
face_connect_vertex = bunny_faces.addConnectivity("face_to_vertex", bunny_vertices, faces, 3)

face_positions = bunny_faces.addAttribute("face_positions", through = face_connect_vertex, source = bunny_vertices_positions)
v0 = face_positions.row(0)
v1 = face_positions.row(1)
v2 = face_positions.row(2)
# compute the area
area = 0.5 * ((v1 - v0).cross(v2 - v0)).norm()
# print(area)
average_area = (sum(area.compute().value.get())) / faces.shape[0]
print(f"average area is: {average_area}")
# check the result
selected_face_ind = 13
selected_face = faces[selected_face_ind]
selected_v0 = vertices[selected_face[0]]
selected_v1 = vertices[selected_face[1]]
selected_v2 = vertices[selected_face[2]]
selected_area = 0.5 * np.linalg.norm(np.cross(np.array(selected_v1) - np.array(selected_v0), np.array(selected_v2) - np.array(selected_v0)))
print(f"Area check: {selected_area}, {area.compute().value.get()[selected_face_ind]}")

# now we define the energy which is the difference of area
bunny_average_area = bunny.addAttribute("avg_area", rows = 1, cols = 1)
bunny_average_area.updateValue([average_area])

energy = bunny_faces.addAttribute("energy", computed_attribute = (area - bunny_average_area) * (area - bunny_average_area))
print(energy)
s0.addEnergy(energy)
s0.addMinimizeTarget([bunny_vertices_positions])
# result = s0.minimizeEnergy()
# print(result)


import pyvista as pv
triangles = np.array(faces)
cells = np.hstack([np.full((triangles.shape[0], 1), 3), triangles])
mesh = pv.PolyData(np.array(vertices), cells)

plotter = pv.Plotter()
plotter.add_mesh(mesh)
plotter.show(interactive_update=True)



total_frames = 0
start = time.time()
iteration = 0
def update_heat():
  global total_frames, start
  change_value = s0.minimizeEnergy()[0]
  new_positions = bunny_vertices_positions.compute().value.get().flatten() - 0.01 * change_value.get()
  bunny_vertices_positions.updateValue(new_positions)
  # Update the mesh points
  mesh.points = new_positions.reshape(-1, 3)

  # Refresh the plotter to reflect the updated mesh
  plotter.update_coordinates(mesh.points, mesh=mesh)
  plotter.render()
  # if iteration % 100 == 0:
  print(f"iteration: {iteration}")
  # print(f"Average area: {(sum(area.compute().value.get())) / faces.shape[0]}, target is: {average_area}")
  print(f"Variance is: {sum((area.compute().value.get() - average_area) ** 2) / faces.shape[0]}")

while True:
  update_heat()
