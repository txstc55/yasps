from yasps import scene
import numpy as np
DT = 0.1
VERTEX_MASS = 0.1
ELASTICITY_CONSTANT = 100000.0
# define the string
NUM_LINE_SEGMENTS = 1000
SEGMENT_LENGTH = 0.01
######################################################
## first we get the bunny from file
######################################################
# first we import bunny obj
f = open('../data/bunny_small.obj', 'r')
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
vertices = np.array(vertices) * 1.0
# first rotate vertices by x axis by 90
R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
vertices = np.dot(vertices, R)
# get the max y coordinates
max_y = np.max(vertices[:, 1])
# get the index with max y
max_y_index = np.argmax(vertices[:, 1])

# translate the bunny top to - NUM_LINE_SEGMENTS * SEGMENT_LENGTH * 2
vertices -= np.array([0, max_y + NUM_LINE_SEGMENTS * SEGMENT_LENGTH * 2, 0])
# translate the bunny top's x to 0
vertices -= np.array([vertices[max_y_index, 0], 0, vertices[max_y_index, 2]])

# ok now we set the starting points of the line
# the last point of the line is the point on bunny
line_vertices = np.array([[0, -(i + 1) * SEGMENT_LENGTH * 2, 0] for i in range(NUM_LINE_SEGMENTS - 1)])


######################################################
## initialize the mesh topology
######################################################
s0 = scene("scene0")
bunny = s0.addMesh("bunny")
bv = bunny.addPrimitive("bunny_vertex", numInstances = vertices.shape[0])
lv = bunny.addPrimitive("line_vertex", numInstances = line_vertices.shape[0])
# we actually don't need to set faces
line_top = bunny.addPrimitive("line_top", numInstances = 1)
line_bottom = bunny.addPrimitive("line_bottom", numInstances = 1)
lines = bunny.addPrimitive("lines", numInstances = NUM_LINE_SEGMENTS - 2)
# add connectivity
l2lv = lines.addConnectivity("to_line_vertex", lv, np.array([[i, i+1] for i in range(NUM_LINE_SEGMENTS - 2)]), 2) # each line segment connects two vertices on the line
lt2lv = line_top.addConnectivity("to_line_vertex", lv, [0], 1) # the top of the line is connected to ceiling
lb2lv = line_bottom.addConnectivity("to_line_vertex", lv, [NUM_LINE_SEGMENTS - 2], 1)
lb2bv = line_bottom.addConnectivity("to_bunny_vertex", bv, [max_y_index], 1) # because we need to connect the line to the bunny

######################################################
## add attributes to mesh
######################################################
from yasps import attribute
roll = bunny.addAttribute("roll", rows = 1, cols = 1)
pitch = bunny.addAttribute("pitch", rows = 1, cols = 1)
yaw = bunny.addAttribute("yaw", rows = 1, cols = 1)
roll.updateValue([0.0])
pitch.updateValue([0.0])
yaw.updateValue([0.0])
# rotation matrix from roll, pitch, yaw
rotation_matrix = attribute.to_array([
  # Row 1
  yaw.cos() * pitch.cos(),
  yaw.cos() * pitch.sin() * roll.sin() - yaw.sin() * roll.cos(),
  yaw.cos() * pitch.sin() * roll.cos() + yaw.sin() * roll.sin(),
  # Row 2
  yaw.sin() * pitch.cos(),
  yaw.sin() * pitch.sin() * roll.sin() + yaw.cos() * roll.cos(),
  yaw.sin() * pitch.sin() * roll.cos() - yaw.cos() * roll.sin(),
  # Row 3
  -pitch.sin(),
  pitch.cos() * roll.sin(),
  pitch.cos() * roll.cos()
], rows = 3, cols = 3)
rot = bunny.addAttribute("rotation_matrix", computed_attribute = rotation_matrix)
# add translation
translation = bunny.addAttribute("translation", rows = 3, cols = 1)
translation.updateValue([0.0, 0.0, 0.0])
# add position to bunny
bvrp = bv.addAttribute("rest_position", rows = 3, cols = 1)
bvrp.updateValue(vertices)
bvp = bv.addAttribute("position", computed_attribute = rot * bvrp + translation) # the position of the bunny is defined by rotation and translation
bvlp = bv.addAttribute("last_position", rows = 3, cols = 1) # for inertia
bvlp.updateValue(vertices)
bvv = bv.addAttribute("velocity", rows = 3, cols = 1) # for inertia
bvv.updateValue([0.0, 0.0, 0.0] * vertices.shape[0])


lvp = lv.addAttribute("position", rows = 3, cols = 1)
lvp.updateValue(line_vertices)
ltp = line_top.addAttribute("top_position", through = lt2lv, source = lvp)
lbbp = line_bottom.addAttribute("bunny_position", through = lb2bv, source = bvp) # the position of the bunny vertex that's connected to the line
lblp = line_bottom.addAttribute("line_position", through = lb2lv, source = lvp) # the position of the line vertex that's connected to the bunny
lp = lines.addAttribute("position", through = l2lv, source = lvp) # the position of the line is defined by the line vertices
# print(lvp.compute().value.get())
# print(lp.compute().value.get())

######################################################
## add energy
######################################################
# we first add the elasticity energy to the lines
lp.reshape(2, 3)
lines.addAttribute("elasticity", computed_attribute = ELASTICITY_CONSTANT * (lp.row(1) - lp.row(0)).dot(lp.row(1) - lp.row(0)))

# then we add elasticity for the top vertex to the ceiling, same elasticity
ltp.reshape(3, 1)
line_top.addAttribute("elasticity", computed_attribute = ELASTICITY_CONSTANT * ltp.dot(ltp))

# then we add elasticity for the bottom vertex to the bunny, same elasticity
lbbp.reshape(3, 1)
lblp.reshape(3, 1)
line_bottom.addAttribute("elasticity", computed_attribute = ELASTICITY_CONSTANT * (lbbp - lblp).dot(lbbp - lblp))

# finally, we need to add inertia to the bunny
def inertia(v0, vel, dt, x, mass):
  # v0 is the position we got before
  # vel is velocity
  # x is the position we are now
  x_target = v0 + vel * dt - attribute.to_array([attribute(float_value = 0.0), attribute(float_value = 9.8 * dt * dt), attribute(float_value = 0.0)], rows = 3, cols = 1)
  return (0.5 * (x - x_target).transpose() * mass * (x - x_target))
# bm = bv.addAttribute("mass", rows = 1, cols = 1)
# bm.updateValue(np.square(np.max(np.abs(vertices[:, 0])) + vertices[:, 0]) * 0.05)
# bm.updateValue(vertices[:, 0])
bm = bv.addAttribute("mass", computed_attribute = bvp[0])
bv.addAttribute("inertia", computed_attribute = inertia(bvlp, bvv, DT, bvp, bm))

# now we add the energy to the scene
s0.addEnergy(lines["elasticity"])
s0.addEnergy(line_top["elasticity"])
s0.addEnergy(line_bottom["elasticity"])
s0.addEnergy(bv["inertia"])
s0.addMinimizeTarget([roll, pitch, yaw, translation, lvp])


######################################################
## plot result
######################################################
import pyvista as pv
bunny_vertices = bvp.compute().value.get()
line_vertices = lvp.compute().value.get()

# Create PyVista PolyData for each vertex array
cloud1 = pv.PolyData(bunny_vertices)
cloud2 = pv.PolyData(line_vertices)

# Create RGB color arrays for each vertex array
colors1 = np.tile([255, 0, 0], (bunny_vertices.shape[0] // 3, 1))  # Red color for all points
colors2 = np.tile([0, 0, 255], (line_vertices.shape[0] // 3, 1))  # Blue color for all points

# Assign RGB colors
cloud1["RGB"] = colors1  # Set RGB colors for set 1
cloud2["RGB"] = colors2  # Set RGB colors for set 2

# Create a PyVista plotter
plotter = pv.Plotter()
plotter.add_mesh(cloud1, scalars="RGB", rgb=True, point_size=2)
plotter.add_mesh(cloud2, scalars="RGB", rgb=True, point_size=2)

# Set the camera position
camera_position = [
    (0, -100, 500),    # Camera position
    (0, 0, 0),    # Focal point (where the camera looks at)
    (0, 1, 0)     # View up direction
]
plotter.camera_position = camera_position

plotter.show(interactive_update=True)
# exit(0)

iteration = 0
bunny_vertices_last = bunny_vertices.copy() # copy just to be safe
while iteration <= 200000:
  # if iteration % 100 == 0:
  #   # save the mesh of the bunny
  #   bunny_vertices = bvp.compute().value.get()
  #   line_vertices = lvp.compute().value.get()
  #   cells = np.hstack([np.full((faces.shape[0], 1), 3), faces])
  #   mesh = pv.PolyData(bunny_vertices, cells)
  #   mesh.save(f"results/bunny_{iteration}.obj")
  #   # save the line as npy file
  #   line_points = np.vstack([np.array([0, 0, 0]).reshape(-1, 3), line_vertices.reshape(-1, 3), bunny_vertices.reshape(-1, 3)[max_y_index].reshape(-1, 3)])
  #   np.save(f"results/line_{iteration}.npy", line_points)

  solution = s0.minimizeEnergy()
  d_roll = solution[0].get()
  d_pitch = solution[1].get()
  d_yaw = solution[2].get()
  d_translation = solution[3].get()
  d_line_vertices = solution[4].get()

  roll.updateValue(roll.value.get() - d_roll * DT)
  pitch.updateValue(pitch.value.get() - d_pitch * DT)
  yaw.updateValue(yaw.value.get() - d_yaw * DT)
  translation.updateValue(translation.value.get() - d_translation * DT)
  lvp.updateValue(lvp.value.get() - d_line_vertices * DT)

  bunny_vertices = bvp.compute().value.get()
  cloud1.points = bunny_vertices.reshape(-1, 3)
  line_vertices = lvp.compute().value.get()
  cloud2.points = line_vertices.reshape(-1, 3)
  plotter.update()
  if (iteration % 1 == 0):
    # update the last position and velocity
    bvlp.updateValue(bunny_vertices)
    bvv.updateValue((bunny_vertices - bunny_vertices_last) / DT)
    bunny_vertices_last = bunny_vertices.copy()

  iteration += 1
