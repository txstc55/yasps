from yasps import scene
def angle(v, w, axis):
  theta = 2.0 * (v.cross(w).dot(axis) / axis.norm()).atan2(v.dot(w) + v.norm() * w.norm())
  return theta
def edgeTheta(q0, q1, q2, q3):
  n0 = (q0 - q2).cross(q1 - q2)
  n1 = (q1 - q3).cross(q0 - q3)
  axis = q1 - q0
  theta = angle(n0, n1, axis)
  return theta
def bending(x, x_init, bendStiff):
  x0 = x.row(0)
  x1 = x.row(1)
  x2 = x.row(2)
  x3 = x.row(3)
  t = edgeTheta(x0, x1, x2, x3)
  x_init0 = x_init.row(0)
  x_init1 = x_init.row(1)
  x_init2 = x_init.row(2)
  x_init3 = x_init.row(3)
  t_init = edgeTheta(x_init0, x_init1, x_init2, x_init3)
  bend_energy = bendStiff * (t - t_init) * (t - t_init) * (x_init1 - x_init0).norm()
  return bend_energy

import numpy as np
import time
# first we import bunny obj
f = open('../data/bunny.obj', 'r')
# read the vertices and faces
vertices = []
faces = []
for line in f:
  line_split = line.split()
  if len(line_split) == 0:
    continue
  if line_split[0] == 'v':
    vertices.append(list(map(float, line[2:].split())))
  if line_split[0] == 'f':
    index = [x.split("//")[0] for x in line[1:].split()]
    faces.append(list(map(int, index)))
f.close()
# make it 0 based
faces = np.array(faces) - 1
vertices = np.array(vertices) * 1000.0
# ok now for each face, we will need to check the edge pairs
edge_to_triangle = {}
for i in range(faces.shape[0]):
  v0 = faces[i, 0]
  v1 = faces[i, 1]
  v2 = faces[i, 2]
  for edge in [(v0, v1), (v1, v2), (v2, v0)]:
    starting_index = edge[0]
    ending_index = edge[1]
    flipped = False
    if starting_index > ending_index:
      starting_index, ending_index = ending_index, starting_index
      flipped = True
    if (starting_index, ending_index) not in edge_to_triangle:
      edge_to_triangle[(starting_index, ending_index)] = []
    edge_to_triangle[(starting_index, ending_index)].append(-i - 1 if flipped else i)

edge_to_triangle_vertices = []
for item in edge_to_triangle:
  if len(edge_to_triangle[item]) == 2:
    indices = [item[0], item[1], -1, -1]
    for triangle_ind in edge_to_triangle[item]:
      # determine the third vertex
      true_ind = triangle_ind
      if triangle_ind < 0:
        true_ind = -triangle_ind - 1
      new_ind = -1
      for i in range(3):
        if faces[true_ind, i] != item[0] and faces[true_ind, i] != item[1]:
          new_ind = faces[true_ind, i]
          break
      if triangle_ind < 0:
        indices[3] = new_ind
      else:
        indices[2] = new_ind
    edge_to_triangle_vertices.append(indices)


s0 = scene("scene0")
bunny = s0.addMesh("bunny")
bunny.addAttribute("bend_stiff", rows = 1, cols = 1)
bunny["bend_stiff"].updateValue([1.0])
bunny_vertices = bunny.addPrimitive("vertices", numInstances = len(vertices))
bunny_vertices_rest_positions = bunny_vertices.addAttribute("rest_position", rows = 1, cols = 3)
bunny_vertices_smoothed_positions = bunny_vertices.addAttribute("smoothed_position", rows = 1, cols = 3)
bunny_vertices_rest_positions.updateValue(vertices)
bunny_vertices_smoothed_positions.updateValue(vertices)
bunny_faces = bunny.addPrimitive("faces", numInstances = len(faces))
face_connect_vertex = bunny_faces.addConnectivity("face_to_vertex", bunny_vertices, faces, 3)
bunny_edge_pairs = bunny.addPrimitive("edge_pairs", numInstances = len(edge_to_triangle_vertices))
edge_pairs_connect_vertex = bunny_edge_pairs.addConnectivity("edge_pairs_to_vertex", bunny_vertices, edge_to_triangle_vertices, 4)

face_positions = bunny_faces.addAttribute("face_positions", through = face_connect_vertex, source = bunny_vertices_smoothed_positions)
edge_pair_rest_positions = bunny_edge_pairs.addAttribute("edge_pair_rest_positions", through = edge_pairs_connect_vertex, source = bunny_vertices_rest_positions)
edge_pair_smoothed_positions = bunny_edge_pairs.addAttribute("edge_pair_smoothed_positions", through = edge_pairs_connect_vertex, source = bunny_vertices_smoothed_positions)



# compute the average areas
v0 = face_positions.row(0)
v1 = face_positions.row(1)
v2 = face_positions.row(2)
# compute the area
area = 0.5 * ((v1 - v0).cross(v2 - v0)).norm()
average_area = (sum(area.compute().value.get())) / faces.shape[0]


area_energy = bunny_faces.addAttribute("area_energy", computed_attribute = 10.0 * (area - float(average_area)) * (area - float(average_area)))
bending_energy = bunny_edge_pairs.addAttribute("bending_energy", computed_attribute = bending(edge_pair_smoothed_positions, edge_pair_rest_positions, 1.0))


s0.addEnergy(area_energy)
s0.addEnergy(bending_energy)
s0.addMinimizeTarget([bunny_vertices_smoothed_positions])


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
weight = 0.1
def update_position():
  global total_frames, start
  change_value = s0.minimizeEnergy()[0].get()
  new_positions = bunny_vertices_smoothed_positions.compute().value.get().flatten() - weight * change_value
  bunny_vertices_smoothed_positions.updateValue(new_positions)
  # Update the mesh points
  mesh.points = new_positions.reshape(-1, 3)

  # Refresh the plotter to reflect the updated mesh
  plotter.update_coordinates(mesh.points, mesh=mesh)
  plotter.render()
  ae = sum(area_energy.compute().value.get().flatten())
  print("Total area energy is:", ae)
  print(f"Variance is: {sum((area.compute().value.get() - average_area) ** 2) / faces.shape[0]}")
  if ae < 35.0:
    # export the mesh
    filename = f"smoothed_mesh/output_bunny_{total_frames}_unsmoothed.obj"
    mesh.save(filename)
    exit()


while total_frames < 1000:
  update_position()
  total_frames += 1
