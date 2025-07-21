from yasps import scene
import numpy as np
import pyvista as pv

s0 = scene("scene0")

# read the obj files
def load_obj(filename):
  vertices = []
  faces = []
  with open(filename, 'r') as f:
    for line in f:
      if line.startswith('v '):  # vertex position
        parts = line.strip().split()
        x, y, z = map(float, parts[1:4])
        vertices.append([x, y, z])
      elif line.startswith('f '):  # face
        parts = line.strip().split()[1:]
        face = []
        for part in parts:
          v_idx = int(part.split('/')[0])  # Only get vertex index
          face.append(v_idx - 1)  # OBJ indices start at 1
        if len(face) == 3:
          faces.append(face)
        elif len(face) == 4:
          # Triangulate quad as two triangles
          faces.append([face[0], face[1], face[2]])
          faces.append([face[0], face[2], face[3]])
        else:
          raise ValueError("Non-triangle/quad face encountered.")

  vertices = np.array(vertices, dtype=np.float32)
  faces = np.array(faces, dtype=np.int32)
  return vertices, faces

###################################################################
# pre process the cylinder object
###################################################################
v, f = load_obj("../data/cylinder_rand.obj")
fixed_vertices = []
moving_vertices = []
modified_indices = []
for i in range(v.shape[0]):
  vertex = v[i]
  if vertex[1] == 0.5 or vertex[1] == -0.5:
    # we move it so the radius is 1.0
    norm = vertex[0]**2 + vertex[2]**2
    vertex[0] = vertex[0] / np.sqrt(norm)
    vertex[2] = vertex[2] / np.sqrt(norm)
    fixed_vertices.append(vertex)
    modified_indices.append(-(len(fixed_vertices)))
  else:
    moving_vertices.append(vertex)
    modified_indices.append(len(moving_vertices) - 1)

new_vertices = np.array(fixed_vertices + moving_vertices, dtype=np.float64)
for i in range(len(modified_indices)):
  if modified_indices[i] < 0:
    modified_indices[i] = -modified_indices[i] - 1
  else:
    modified_indices[i] = modified_indices[i] + len(fixed_vertices)

if len(set(modified_indices)) != len(modified_indices):
  raise ValueError("Modified indices are not unique, something went wrong in the preprocessing.")
new_faces = []
for face in f:
  new_faces.append([modified_indices[face[0]], modified_indices[face[1]], modified_indices[face[2]]])

new_faces = np.array(new_faces, dtype=np.uint32)
faces_flat = np.hstack(
    [np.full((new_faces.shape[0], 1), 3, dtype=np.uint32), new_faces]
).flatten()


###################################################################
# construct the mesh with fixed and moving vertices
###################################################################
cylinder = s0.addMesh("cylinder")
fv = cylinder.addPrimitive("fixed_vertices", numInstances = len(fixed_vertices))
fvp = fv.addAttribute("position", rows = 3, cols = 1)
fvp.updateValue(np.array(fixed_vertices))

mv = cylinder.addPrimitive("moving_vertices", numInstances = len(moving_vertices))
mvp = mv.addAttribute("position", rows = 3, cols = 1)
mvp.updateValue(np.array(moving_vertices))


###################################################################
# add the primitive union
###################################################################
v = cylinder.addPrimitiveUnion("vertices", [fv, mv])
vp = v.addAttribute("position")


###################################################################
# add faces to it
###################################################################
f = cylinder.addPrimitive("faces", numInstances = new_faces.shape[0])
f2v = f.addConnectivity("faces_to_vertices", v, new_faces, 3)
fp = f.addAttribute("position", through = f2v)

# add computation for the aream of the face
v0 = fp.row(0)
v1 = fp.row(1)
v2 = fp.row(2)
area = ((v1 - v0).cross(v2 - v0)).norm()
f.addAttribute("area", computed_attribute = area)

s0.addEnergy(area)
s0.addMinimizeTarget([mvp])

###################################################################
# update and plot
###################################################################
# Create the mesh
mesh = pv.PolyData(new_vertices, faces_flat)
# Plot it
plotter = pv.Plotter()
plotter.add_mesh(mesh, color="lightblue", show_edges=True)
plotter.show(interactive_update=True)

DT = 0.01
for _ in range(10000):
  dmvp = s0.minimizeEnergy(tolerance = 1e-3)[0]
  mvp.updateValue(mvp.value - DT * dmvp)
  new_positions = vp.compute().value.get().reshape(-1, 3)
  mesh.points = new_positions
  plotter.update_coordinates(mesh.points, mesh=mesh)
  plotter.render()
  plotter.update()
