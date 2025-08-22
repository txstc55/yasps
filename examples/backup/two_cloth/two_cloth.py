from yasps import scene
from yasps import attribute
import numpy as np
NUM_SEGMENTS = 100

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

def generate_cloth_mesh(length_of_square, num_subdivisions):
  num_vertices_per_side = num_subdivisions + 1
  # Compute step size along one edge
  step = length_of_square / num_subdivisions
  # Generate vertex positions
  vertices = []
  for j in range(num_vertices_per_side):
    for i in range(num_vertices_per_side):
      x = i * step - length_of_square / 2
      y = 0.0
      z = j * step - length_of_square / 2
      vertices.append((x, y, z))
  faces = []
  for j in range(num_subdivisions):
    for i in range(num_subdivisions):
      # Calculate the vertex indices
      v0 = j * num_vertices_per_side + i
      v1 = v0 + 1
      v2 = (j + 1) * num_vertices_per_side + i
      v3 = v2 + 1
      faces.append((v2, v1, v0))
      faces.append((v3, v1, v2))
  return vertices, faces

def compute_rest_shape(v0, v1, v2):
  v01 = v1 - v0
  v02 = v2 - v0
  normal = v01.cross(v02)
  normal = normal / normal.norm()
  target = attribute.to_array([0, 1, 0], rows = 1, cols = 3)
  vec = normal.cross(target)
  cos = normal.dot(target)
  rotation = attribute.identity(3)
  cross_vec = attribute.to_array([0, -vec[2], vec[1], vec[2], 0, -vec[0], -vec[1], vec[0], 0], rows = 3, cols = 3)
  rotation = rotation + cross_vec + cross_vec * cross_vec / (1 + cos)
  rotate_uv0 = rotation * v0.transpose()
  rotate_uv1 = rotation * v1.transpose()
  rotate_uv2 = rotation * v2.transpose()
  uv0 = attribute.to_array([rotate_uv0[0], rotate_uv0[2]], rows = 1, cols = 2)
  uv1 = attribute.to_array([rotate_uv1[0], rotate_uv1[2]], rows = 1, cols = 2)
  uv2 = attribute.to_array([rotate_uv2[0], rotate_uv2[2]], rows = 1, cols = 2)
  uv1_minus_uv0 = uv1 - uv0
  uv2_minus_uv0 = uv2 - uv0
  M = attribute.to_array([uv1_minus_uv0[0], uv2_minus_uv0[0], uv1_minus_uv0[1], uv2_minus_uv0[1]], rows = 2, cols = 2)
  return M


def baraff_witkin_energy(x_init, x, stretchS, shearS, thickness, dt):
  anisotropic_a = attribute.to_array([1, 0], rows = 1, cols = 2)
  anisotropic_b = attribute.to_array([0, 1], rows = 1, cols = 2)
  x10 = x.row(1) - x.row(0)
  x20 = x.row(2) - x.row(0)
  F = attribute.to_array([x10[0], x10[1], x10[2], x20[0], x20[1], x20[2]], rows = 2, cols = 3)
  F = F.transpose()
  F = F * compute_rest_shape(x_init.row(0), x_init.row(1), x_init.row(2)).inverse()
  I6 = (anisotropic_a * F.transpose() * F * anisotropic_b.transpose())
  shear_energy = I6 * I6
  I5u = (F * anisotropic_a.transpose()).norm()
  I5v = (F * anisotropic_b.transpose()).norm()
  ucoeff = 1.0
  vcoeff = 1.0
  stretch_energy = ucoeff * (I5u - 1.0) * (I5u - 1.0) + vcoeff * (I5v - 1.0) * (I5v - 1.0)
  v01 = x_init.row(1) - x_init.row(0)
  v02 = x_init.row(2) - x_init.row(0)
  area = thickness * v01.cross(v02).norm() / 2.0
  return (stretchS * stretch_energy + shearS * shear_energy) * area * dt * dt

def inertia(v0, vel, dt, x, mass):
  x_target = v0 + vel * dt - attribute.to_array([0.0, 9.8 * dt * dt, 0.0], rows = 3, cols = 1)
  return (0.5 * (x - x_target).transpose() * mass * (x - x_target))

def point_sphere_collision(v0, center, radius, dHat, kappa):
  v1 = v0 - center
  v1 = (v1 / v1.norm()) * radius + center
  d = ((v1 - v0).dot(v1 - v0)).sqrt()
  d = attribute.select(d < attribute(float_value = dHat), d, attribute(float_value = dHat))
  b = -((d - dHat) * (d - dHat)) * (d / dHat).log()
  return kappa * b



def closest_point_on_triangle(pt, v0, v1, v2):
  v0p = pt - v0
  v0v1 = v1 - v0
  v0v2 = v2 - v0
  # Dot products for barycentric
  dot00 = v0v2.dot(v0v2)
  dot01 = v0v2.dot(v0v1)
  dot02 = v0v2.dot(v0p)
  dot11 = v0v1.dot(v0v1)
  dot12 = v0v1.dot(v0p)
  denom = dot00 * dot11 - dot01 * dot01
  alpha = (dot11 * dot02 - dot01 * dot12) / denom
  beta  = (dot00 * dot12 - dot01 * dot02) / denom
  alpha = attribute.select(alpha < attribute(float_value = 0.0), attribute(float_value = 0.0), alpha)
  alpha = attribute.select(alpha > attribute(float_value = 1.0), attribute(float_value = 1.0), alpha)
  beta = attribute.select(beta < attribute(float_value = 0.0), attribute(float_value = 0.0), beta)
  beta = attribute.select(alpha + beta <= attribute(float_value = 1.0), beta, 1.0 - alpha)
  return v0 + alpha * v0v2 + beta * v0v1

def triangle_sphere_collision(v0, v1, v2, center, radius, dHat, kappa):
  # 1) Get closest point on triangle to the sphere center
  cp = closest_point_on_triangle(center, v0, v1, v2)
  to_cp = cp - center
  to_cp = to_cp / to_cp.norm() * radius + center
  d = ((cp - to_cp).dot(cp - to_cp)).sqrt()
  d = attribute.select(d < attribute(float_value = dHat), d, attribute(float_value = dHat))
  b = -((d - dHat) * (d - dHat)) * (d / dHat).log()
  return kappa * b


###################################################
## initialize vertices and faces
## and also edge to the 4 vertices for bending energy
###################################################
vertices, faces = generate_cloth_mesh(6.0, NUM_SEGMENTS)
vertices = np.array(vertices) + np.array([0.0, 3.0, 0.0])
faces = np.array(faces)
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

###################################################
## initialize mesh
###################################################
s0 = scene("scene0")
s0.addAttribute("sphere_center", rows = 3, cols = 1)
s0.addAttribute("sphere_radius", rows = 1, cols = 1)
s0["sphere_center"].updateValue([0.0, 0.0, 0.0])
s0["sphere_radius"].updateValue([1.0])

cloth0 = s0.addMesh("cloth0")
cloth1 = s0.addMesh("cloth1")
DT = 0.1

v0 = cloth0.addPrimitive("vertices", numInstances = vertices.shape[0])
f0 = cloth0.addPrimitive("faces", numInstances = faces.shape[0])
e0 = cloth0.addPrimitive("edge_pairs", numInstances = len(edge_to_triangle_vertices))

v1 = cloth1.addPrimitive("vertices", numInstances = vertices.shape[0])
f1 = cloth1.addPrimitive("faces", numInstances = faces.shape[0])
e1 = cloth1.addPrimitive("edge_pairs", numInstances = len(edge_to_triangle_vertices))

# add connectivity
face_connect_vertex0 = f0.addConnectivity("face_to_vertex", v0, faces, 3)
edge_connect_vertex0 = e0.addConnectivity("edge_to_vertex", v0, edge_to_triangle_vertices, 4)

face_connect_vertex1 = f1.addConnectivity("face_to_vertex", v1, faces, 3)
edge_connect_vertex1 = e1.addConnectivity("edge_to_vertex", v1, edge_to_triangle_vertices, 4)

# add position attribute to vertices
vrp0 = v0.addAttribute("rest_position", rows = 3, cols = 1)
vrp0.updateValue(vertices)
vp0 = v0.addAttribute("position", rows = 3, cols = 1)
vp0.updateValue(vertices)
vm0 = v0.addAttribute("mass", rows = 1, cols = 1)
vm0.updateValue(np.ones(vertices.shape[0]) * 10.0 / vertices.shape[0])
vv0 = v0.addAttribute("velocity", rows = 3, cols = 1)
vv0.updateValue(np.zeros(vertices.shape))
vlp0 = v0.addAttribute("last_position", rows = 3, cols = 1)
vlp0.updateValue(vp0.value.get())


vrp1 = v1.addAttribute("rest_position", rows = 3, cols = 1)
vrp1.updateValue(vertices + np.array([0, 3, 0]))
vp1 = v1.addAttribute("position", rows = 3, cols = 1)
vp1.updateValue(vertices + np.array([0, 3, 0]))
vm1 = v1.addAttribute("mass", rows = 1, cols = 1)
vm1.updateValue(np.ones(vertices.shape[0]) * 20.0 / vertices.shape[0])
vv1 = v1.addAttribute("velocity", rows = 3, cols = 1)
vv1.updateValue(np.zeros(vertices.shape))
vlp1 = v1.addAttribute("last_position", rows = 3, cols = 1)
vlp1.updateValue(vp0.value.get())

# join values to each triangle
frp0 = f0.addAttribute("rest_position", through = face_connect_vertex0, source = vrp0)
fp0 = f0.addAttribute("position", through = face_connect_vertex0, source = vp0)

frp1 = f1.addAttribute("rest_position", through = face_connect_vertex1, source = vrp1)
fp1 = f1.addAttribute("position", through = face_connect_vertex1, source = vp1)

# join values to each edge
erp0 = e0.addAttribute("rest_position", through = edge_connect_vertex0, source = vrp0)
ep0 = e0.addAttribute("position", through = edge_connect_vertex0, source = vp0)

erp1 = e1.addAttribute("rest_position", through = edge_connect_vertex1, source = vrp1)
ep1 = e1.addAttribute("position", through = edge_connect_vertex1, source = vp1)

# add attribute to faces
stretch0 = f0.addAttribute("stretch0", rows = 1, cols = 1)
stretch0.updateValue(np.ones(len(faces)) * 10000.0)
shear0 = f0.addAttribute("shear0", rows = 1, cols = 1)
shear0.updateValue(np.ones(len(faces)) * 1000.0)

stretch1 = f1.addAttribute("stretch0", rows = 1, cols = 1)
stretch1.updateValue(np.ones(len(faces)) * 30000.0)
shear1 = f1.addAttribute("shear0", rows = 1, cols = 1)
shear1.updateValue(np.ones(len(faces)) * 4000.0)

###################################################
## add energy
###################################################
bw0 = f0.addAttribute("baraff_witkin", computed_attribute = baraff_witkin_energy(frp0, fp0, 1000, 100, 1.0, DT))
triangle_collision0 = f0.addAttribute("triangle_collision0", computed_attribute = triangle_sphere_collision(fp0.row(0), fp0.row(1), fp0.row(2), s0["sphere_center"].transpose(), s0["sphere_radius"], 0.1, 1.0))

bw1 = f1.addAttribute("baraff_witkin", computed_attribute = baraff_witkin_energy(frp1, fp1, 1000, 100, 1.0, DT))
triangle_collision1 = f1.addAttribute("triangle_collision0", computed_attribute = triangle_sphere_collision(fp1.row(0), fp1.row(1), fp1.row(2), s0["sphere_center"].transpose(), s0["sphere_radius"], 0.1, 1.0))

point_collision0 = v0.addAttribute("point_collision0", computed_attribute = point_sphere_collision(vp0, s0["sphere_center"], s0["sphere_radius"], 0.1, 1.0))
inertia_energy0 = v0.addAttribute("inertia0", computed_attribute = inertia(vlp0, vv0, DT, vp0, vm0))
bending_energy0 = e0.addAttribute("bending_energy0", computed_attribute = bending(ep0, erp0, 0.001))

point_collision1 = v1.addAttribute("point_collision0", computed_attribute = point_sphere_collision(vp1, s0["sphere_center"], s0["sphere_radius"], 0.1, 1.0))
inertia_energy1 = v1.addAttribute("inertia0", computed_attribute = inertia(vlp1, vv1, DT, vp1, vm1))
bending_energy1 = e1.addAttribute("bending_energy0", computed_attribute = bending(ep1, erp1, 0.001))

s0.addEnergy(point_collision0)
s0.addEnergy(triangle_collision0)
s0.addEnergy(bw0)
s0.addEnergy(inertia_energy0)
s0.addEnergy(bending_energy0)

s0.addEnergy(point_collision1)
s0.addEnergy(triangle_collision1)
s0.addEnergy(bw1)
s0.addEnergy(inertia_energy1)
s0.addEnergy(bending_energy1)

s0.addMinimizeTarget([vp0, vp1])

###################################################
## visualize the mesh
###################################################
import pyvista as pv
triangles = np.array(faces)
cells = np.hstack([np.full((triangles.shape[0], 1), 3), triangles])
mesh0 = pv.PolyData(vp0.value.get().reshape(-1, 3), cells)
mesh1 = pv.PolyData(vp1.value.get().reshape(-1, 3), cells)


plotter = pv.Plotter()
plotter.add_mesh(mesh0)
plotter.add_mesh(mesh1)
sphere_mesh = pv.Sphere(radius=1.0, center=(0,0,0), phi_resolution=30, theta_resolution=30)
plotter.add_mesh(sphere_mesh, color='cyan')
# Set the camera position
camera_position = [
  (0, 30, 50),    # Camera position
  (0, 0, 0),    # Focal point (where the camera looks at)
  (0, 1, 0)     # View up direction
]
plotter.camera_position = camera_position
plotter.show(interactive_update=True)

iteration = 0
cloth_vertices_last0 = vp0.value.copy()
cloth_vertices_last1 = vp1.value.copy()
while iteration < 1000:
  vlp0.updateValue(cloth_vertices_last0, deepCopy=True)
  vlp1.updateValue(cloth_vertices_last1, deepCopy=True)
  solution = s0.minimizeEnergy(tolerance = 1e-6)
  dp0 = solution[0]
  dp1 = solution[1]
  vp_new0 = vp0.value - dp0 * DT
  vp_new1 = vp1.value - dp1 * DT

  vp0.updateValue(vp_new0, deepCopy = True)
  vp1.updateValue(vp_new1, deepCopy = True)

  mesh0.points = vp_new0.get().reshape(-1, 3)
  mesh1.points = vp_new1.get().reshape(-1, 3)

  plotter.update_coordinates(mesh0.points, mesh=mesh0)
  plotter.update_coordinates(mesh1.points, mesh=mesh1)
  plotter.render()
  plotter.update()
  cloth_vertices0 = vp_new0
  cloth_vertices1 = vp_new1

  # update velocities
  if iteration % 1 == 0:
    vv0.updateValue((cloth_vertices0 - cloth_vertices_last0) / DT, deepCopy = True) # damp the velocity a bit
    vv1.updateValue((cloth_vertices1 - cloth_vertices_last1) / DT, deepCopy = True) # damp the velocity a bit
    cloth_vertices_last0 = cloth_vertices0.copy()
    cloth_vertices_last1 = cloth_vertices1.copy()
  iteration += 1
  print("Iteration:", iteration)
  # if iteration == 999:
  #   mesh.points = vp_new0.get().reshape(-1, 3)
  #   plotter.update_coordinates(mesh.points, mesh=mesh)
  #   plotter.render()
  #   plotter.update()

# # export the final mesh
# mesh.save(f"cloth_out/cloth_{faces.shape[0]}.obj")
