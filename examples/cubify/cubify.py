from yasps import scene
import numpy as np
from yasps import attribute

############################################
# initialize with real data
############################################
f = open('../data/bunny_small.obj', 'r')
vertices = []
faces = []
for line in f:
  line_split = line.split()
  if line_split[0] == 'v':
    vertices.append(list(map(float, line[2:].split())))
  elif line_split[0] == 'f':
    faces.append([int(item.split("//")[0]) - 1 for item in line_split[1:]])
f.close()

vertices = np.array(vertices, dtype = np.float64)
faces = np.array(faces)

# we initially rotate the mesh by some degrees
angle_degrees = 10
theta = np.radians(angle_degrees)
cos_theta = np.cos(theta)
sin_theta = np.sin(theta)
R_x = np.array([
    [1, 0,         0        ],
    [0, cos_theta, -sin_theta],
    [0, sin_theta, cos_theta ]
])
# vertices = vertices[:4]
# print(vertices.tolist())
# faces = np.array([[0, 1, 2], [0, 3, 1]])

# for the cubifying example
# because we don't support dynamic for loop, instead we will expand the relationships
# the two relationships we need are
# vertex to edge (for ARAP)
# vertex to face (for the second term in the paper)
# and because we need to compute the cotangent weight for each edge
# we additionally need
# an edge to triangle relationship (or edge to neighboring vertices relationshio)
# to compute the contangent weight
# the minization will minimize the angles for rotation for ARAP, which is performed
# on the vertex to edge relationshio
# and the regularization term, which is performed
# on the vertex to triangle relationship

vertex_to_face = [[] for i in range(len(vertices))]
for i in range(len(faces)):
  v0, v1, v2 = faces[i]
  vertex_to_face[v0].append(i)
  vertex_to_face[v1].append(i)
  vertex_to_face[v2].append(i)

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

# each edge will correspond to 4 vertices
# the two vertices of the edge, and the two vertices of the two triangles
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
v = bunny.addPrimitive("vertices", numInstances = len(vertices))
v.addAttribute("position", rows = 3, cols = 1)
v.addAttribute("rest_position", rows = 3, cols = 1)

v["rest_position"].updateValue(vertices)

position_new = vertices @ R_x.T
position_new[10, 0] += 2.0
v["position"].updateValue(position_new)
# use quaternions to represent the rotation
w = v.addAttribute("w", rows = 1, cols = 1)
x = v.addAttribute("x", rows = 1, cols = 1)
y = v.addAttribute("y", rows = 1, cols = 1)
z = v.addAttribute("z", rows = 1, cols = 1)
w.updateValue([1.0] * len(vertices))
x.updateValue([0.0] * len(vertices))
y.updateValue([0.0] * len(vertices))
z.updateValue([0.0] * len(vertices))

v.addAttribute("rotation_matrix", computed_attribute = attribute.to_array([
        1 - 2*(y * y + z * z),     2*(x*y - w*z),         2*(x*z + w*y),
        2*(x*y + w*z),           1 - 2*(x * x + z * z),   2*(y*z - w*x),
        2*(x*z - w*y),           2*(y*z + w*x),         1 - 2*(x * x + y * y)], rows = 3, cols = 3))

# v.addAttribute("rotation_matrix", computed_attribute = attribute.to_array([
#   pitch.cos(), roll.sin() * pitch.sin(), roll.cos() * pitch.sin(),
#   attribute(float_value = 0.0), roll.cos(), -roll.sin(),
#   -pitch.sin(), roll.sin() * pitch.cos(), roll.cos() * pitch.cos()
# ], rows = 3, cols = 3))


f = bunny.addPrimitive("faces", numInstances = faces.shape[0])
face_to_vertex = f.addConnectivity('face_to_vertex', v, faces, 3)
f.addAttribute("rest_position", through = face_to_vertex)
f.addAttribute("position", through = face_to_vertex)
f.addAttribute("area", computed_attribute = (f["rest_position"].row(1) - f["rest_position"].row(0)).cross(f["rest_position"].row(2) - f["rest_position"].row(0)).norm() / 2.0)
# now we accumulate the area of the faces to the vertices
v2f = v.addConnectivity('vertex_to_face', f, vertex_to_face, 0)
v.addAttribute("area", through = v2f, operation = "SUM", source = f["area"])
v.addAttribute("area_constant", rows = 1, cols = 1)
v["area_constant"].updateValue(v["area"].compute().value, deepCopy = True) # update the value to be constant
f.addAttribute("total_area", through = face_to_vertex, source = v["area_constant"])
rot = f.addAttribute("rotation_matrix", through = face_to_vertex)
f.addAttribute("rot0", computed_attribute = attribute.to_array([rot[i] for i in range(0, 9)], rows = 3, cols = 3))
f.addAttribute("rot1", computed_attribute = attribute.to_array([rot[i] for i in range(9, 18)], rows = 3, cols = 3))
f.addAttribute("rot2", computed_attribute = attribute.to_array([rot[i] for i in range(18, 27)], rows = 3, cols = 3))
# get the normal of the face
f.addAttribute("normal", computed_attribute = (f["position"].row(1) - f["position"].row(0)).cross(f["position"].row(2) - f["position"].row(0)))
# add the regularization energy
rotated_v0 = f["rot0"] * f["normal"]
rotated_v1 = f["rot1"] * f["normal"]
rotated_v2 = f["rot2"] * f["normal"]
weighted_v0 = (rotated_v0[0].abs() + rotated_v0[1].abs() + rotated_v0[2].abs()) * f["area"]
weighted_v1 = (rotated_v1[0].abs() + rotated_v1[1].abs() + rotated_v1[2].abs()) * f["area"]
weighted_v2 = (rotated_v2[0].abs() + rotated_v2[1].abs() + rotated_v2[2].abs()) * f["area"]
f.addAttribute("cubify_energy", computed_attribute = 0.1 * (weighted_v0 + weighted_v1 + weighted_v2))


# add the edge primitive
e = bunny.addPrimitive("edges", numInstances = len(edge_to_triangle_vertices))
edge_to_triangle_vertex = e.addConnectivity('edge_to_triangle_vertex', v, edge_to_triangle_vertices, 4)
er = e.addAttribute("four_vertices_rest_position", through = edge_to_triangle_vertex, source = v["rest_position"]) # because it's supposed to be a 4x3 matrix
ep = e.addAttribute("four_vertices_position", through = edge_to_triangle_vertex, source = v["position"])
er.reshape(4, 3)
ep.reshape(4, 3)

# compute the contangent weight
def contangent_weight(v0, v1, v2):
  a = v0 - v2
  b = v1 - v2
  # compute the angle between
  cot_weight = a.dot(b) / a.cross(b).norm()
  return cot_weight
e.addAttribute("cotangent_weight", computed_attribute = (contangent_weight(er.row(0), er.row(1), er.row(2)) + contangent_weight(er.row(0), er.row(1), er.row(3))) / 4.0)
# print(e["cotangent_weight"].compute().value.get())

# now we also add the edge to two vertex connectivity
edge_to_vertex0 = e.addConnectivity('edge_to_vertex0', v, [(item[0]) for item in edge_to_triangle_vertices], 1)
edge_to_vertex1 = e.addConnectivity('edge_to_vertex1', v, [(item[1]) for item in edge_to_triangle_vertices], 1)
edge_position0 = e.addAttribute("position0", through = edge_to_vertex0, source = v["position"])
edge_rest_position0 = e.addAttribute("rest_position0", through = edge_to_vertex0, source = v["rest_position"])
edge_position1 = e.addAttribute("position1", through = edge_to_vertex1, source = v["position"])
edge_rest_position1 = e.addAttribute("rest_position1", through = edge_to_vertex1, source = v["rest_position"])
rotation0 = e.addAttribute("rotation0", through = edge_to_vertex0, source = v["rotation_matrix"])
rotation0.reshape(3, 3)
rotation1 = e.addAttribute("rotation1", through = edge_to_vertex1, source = v["rotation_matrix"])
rotation1.reshape(3, 3)

edge_rest = (edge_rest_position1 - edge_rest_position0).transpose()
edge = (edge_position1 - edge_position0).transpose()

e.addAttribute("arap_energy", computed_attribute = 0.1 * e["cotangent_weight"] * ((rotation0 * edge_rest - edge).norm() + (rotation1 * edge_rest - edge).norm()))

# add bending energy
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
  # bend_energy = bendStiff * (x_init1 - x_init0).norm()
  bend_energy = bendStiff * (t - t_init) * (t - t_init) * (x_init1 - x_init0).norm() + bendStiff * ((x3 - x2).norm() - (x_init3 - x_init2).norm()) * ((x3 - x2).norm() - (x_init3 - x_init2).norm())
  bend_energy = bendStiff * (t - t_init) * (t - t_init) * (x_init1 - x_init0).norm()
  return bend_energy

bending_energy = e.addAttribute("bending_energy", computed_attribute = bending(ep, er, 10.0))


s0.addEnergy(e["arap_energy"])
# s0.addEnergy(f["cubify_energy"])
# s0.addEnergy(e["bending_energy"])

s0.addMinimizeTarget([v["position"], w, x, y, z])
result = s0.minimizeEnergy()


import pyvista as pv
triangles = np.array(faces)
cells = np.hstack([np.full((triangles.shape[0], 1), 3), triangles])
mesh = pv.PolyData(v["position"].value.get().reshape(-1, 3), cells)

plotter = pv.Plotter()
plotter.add_mesh(mesh)
plotter.show(interactive_update=True)
total_frames = 0
iteration = 0
weight = 0.01
def update_position():
  global total_frames, start, iteration
  change_value = s0.minimizeEnergy()
  new_positions = v["position"].value.get().flatten() - weight * change_value[0].get().flatten()
  v["position"].updateValue(new_positions)
  mesh.points = new_positions.reshape(-1, 3)
  new_w = w.value.get() - weight * change_value[1].get()
  new_x = x.value.get() - weight * change_value[2].get()
  new_y = y.value.get() - weight * change_value[3].get()
  new_z = z.value.get() - weight * change_value[4].get()
  quaternions = np.stack((new_w, new_x, new_y, new_z), axis=-1)  # Shape: (N, 4)
  norms = np.linalg.norm(quaternions, axis=1)  # Shape: (N,)
  norms[norms == 0] = 1
  # Reshape norms to make it broadcastable for division
  norms = norms[:, np.newaxis]  # Shape: (N, 1)
  # Normalize the quaternions
  normalized_quaternions = quaternions / norms  # Shape: (N, 4)
  w.updateValue(normalized_quaternions[:, 0])
  x.updateValue(normalized_quaternions[:, 1])
  y.updateValue(normalized_quaternions[:, 2])
  z.updateValue(normalized_quaternions[:, 3])

  # we need to normalize the quaternion

  # Update the mesh points

  iteration += 1
  total_frames += 1
  # Refresh the plotter to reflect the updated mesh
  print("Arap energy")
  print(sum(e["arap_energy"].compute().value.get()))
  print("Cubify energy")
  print(sum(f["cubify_energy"].compute().value.get()))
  plotter.update_coordinates(mesh.points, mesh=mesh)
  plotter.render()


while iteration < 1000:
  update_position()
  # if total_frames % 1000 == 0:
  #   weight *= 0.9
  if total_frames % 10 == 0:
    # export the mesh
    filename = f"../data/output_bunny_small_cubify_{total_frames}.obj"
    mesh.save(filename)
