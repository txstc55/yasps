from yasps import scene
from yasps import attribute
import numpy as np
NUM_SEGMENTS = 10

def generate_cloth_mesh(length_of_square, num_subdivisions):
  num_vertices_per_side = num_subdivisions + 1
  # Compute step size along one edge
  step = length_of_square / num_subdivisions
  # Generate vertex positions
  vertices = []
  for j in range(num_vertices_per_side):
    for i in range(num_vertices_per_side):
      x = i * step - length_of_square / 2
      y = x / 2.0
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

def floor_collision(v0, dHat, offset, kappa):
  d = (((attribute.to_array([0.0, 1.0, 0.0], rows = 3, cols = 1).dot(v0)) - attribute(float_value = offset)) * ((attribute.to_array([0.0, 1.0, 0.0], rows = 3, cols = 1).dot(v0)) - attribute(float_value = offset))).sqrt() # ok i know the squre then sqrt isn't making sense, but this is because in the current code, the select operator's correspondance is hard to check. If a select operator has two path that both leads to a constant value, is the selec operator just a constant mat? It obviously is not, but if I mark it as not a constant mat, then what's the correspondance? There is no correspondance because the output doesn't need any primitive. This is just a reminder to redo the check heritage function or at least add more special cases.
  d = attribute.select(d < attribute(float_value = dHat), d, attribute(float_value = dHat))
  b = -((d - dHat) * (d - dHat)) * (d / dHat).log()
  return kappa * b


###################################################
## initialize vertices and faces
###################################################
vertices, faces = generate_cloth_mesh(6.0, NUM_SEGMENTS)
vertices = np.array(vertices) + np.array([0.0, 3.0, 0.0])
faces = np.array(faces)


###################################################
## initialize mesh
###################################################
s0 = scene("scene0")
cloth = s0.addMesh("cloth")
DT = 0.1
# add the sphere center and radius
cloth.addAttribute("sphere_center", rows = 3, cols = 1)
cloth["sphere_center"].updateValue([0.0, 0.0, 0.0])
cloth.addAttribute("sphere_radius", rows = 1, cols = 1)
cloth["sphere_radius"].updateValue([1.0])

v = cloth.addPrimitive("vertices", numInstances = vertices.shape[0])
f = cloth.addPrimitive("faces", numInstances = faces.shape[0])
# add connectivity
face_connect_vertex = f.addConnectivity("face_to_vertex", v, faces, 3)
# add position attribute to vertices
vrp = v.addAttribute("rest_position", rows = 3, cols = 1)
vrp.updateValue(vertices)
vp = v.addAttribute("position", rows = 3, cols = 1)
vp.updateValue(vertices)
vm = v.addAttribute("mass", rows = 1, cols = 1)
vm.updateValue(np.ones(vertices.shape[0]) * 100.0 / vertices.shape[0])
vv = v.addAttribute("velocity", rows = 3, cols = 1)
vv.updateValue(np.zeros(vertices.shape))
vlp = v.addAttribute("last_position", rows = 3, cols = 1)
vlp.updateValue(vp.value.get())

# gather values to each triangle
frp = f.addAttribute("rest_position", through = face_connect_vertex, source = vrp)
fp = f.addAttribute("position", through = face_connect_vertex, source = vp)
# add attribute to faces
stretch = f.addAttribute("stretch", rows = 1, cols = 1)
stretch.updateValue(np.ones(len(faces)) * 10000.0)
shear = f.addAttribute("shear", rows = 1, cols = 1)
shear.updateValue(np.ones(len(faces)) * 1000.0)

###################################################
## add energy
###################################################
bw = f.addAttribute("baraff_witkin", computed_attribute = baraff_witkin_energy(frp, fp, 10000, 1000, 0.1, DT))
floor_collision_energy = v.addAttribute("floor_collision_energy", computed_attribute = floor_collision(vp, 0.2 , -2.0, 1.0))
inertia_energy = v.addAttribute("inertia", computed_attribute = inertia(vlp, vv, DT, vp, vm))
s0.addEnergy(floor_collision_energy)
s0.addEnergy(bw)
s0.addEnergy(inertia_energy)
s0.addMinimizeTarget([vp])

###################################################
## visualize the mesh
###################################################
import pyvista as pv
triangles = np.array(faces)
cells = np.hstack([np.full((triangles.shape[0], 1), 3), triangles])
mesh = pv.PolyData(vp.value.get().reshape(-1, 3), cells)

plotter = pv.Plotter()
plotter.add_mesh(mesh)
# sphere_mesh = pv.Sphere(radius=1.0, center=(0,0,0), phi_resolution=30, theta_resolution=30)
# plotter.add_mesh(sphere_mesh, color='cyan')
# Set the camera position
camera_position = [
  (0, 30, 50),    # Camera position
  (0, 0, 0),    # Focal point (where the camera looks at)
  (0, 1, 0)     # View up direction
]
plotter.camera_position = camera_position
plotter.show(interactive_update=True)

iteration = 0
cloth_vertices_last = vp.value.copy()
while iteration < 1000:
  vlp.updateValue(cloth_vertices_last, deepCopy=True)
  solution = s0.minimizeEnergy(tolerance = 1e-6)
  dp = solution[0]
  vp_new = vp.value - dp * DT
  vp.updateValue(vp_new, deepCopy = True)
  mesh.points = vp_new.get().reshape(-1, 3)
  plotter.update_coordinates(mesh.points, mesh=mesh)
  plotter.render()
  plotter.update()
  cloth_vertices = vp_new
  # update velocities
  if iteration % 1 == 0:
    vv.updateValue((cloth_vertices - cloth_vertices_last) / DT, deepCopy = True) # damp the velocity a bit
    bunny_vertices_last = cloth_vertices.copy()
  iteration += 1
  print("Iteration:", iteration)
  if iteration == 999:
    mesh.points = vp_new.get().reshape(-1, 3)
    plotter.update_coordinates(mesh.points, mesh=mesh)
    plotter.render()
    plotter.update()

# export the final mesh
mesh.save(f"cloth_fall_out/cloth_{faces.shape[0]}.obj")
