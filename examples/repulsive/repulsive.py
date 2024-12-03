from yasps.scene import scene
from yasps.attribute import attribute
from yasps.autodiff import autodiff
import numpy as np
import math
np.random.seed(13)
NUM_POINTS = 500
NUM_LOOPS = 5
def generate_points_near_equator(num_points, delta_phi_deg=5):
  theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False, dtype=np.float64)
  delta_phi_rad = np.radians(delta_phi_deg)
  phi = np.random.uniform(-delta_phi_rad, delta_phi_rad, num_points)
  x = np.cos(phi) * np.cos(theta)
  y = np.cos(phi) * np.sin(theta)
  z = np.sin(phi)
  # normalize
  norm = np.linalg.norm(np.vstack((x, y, z)).T, axis=1)
  points = np.vstack((x, y, z)).T / norm[:, None]
  return points

vertex_positions_value = np.empty((NUM_POINTS * NUM_LOOPS, 3), dtype=np.float64)
for i in range(NUM_LOOPS):
  vertex_positions = generate_points_near_equator(NUM_POINTS) * 0.3
  vertex_positions = vertex_positions - np.array([0.3, 0, 0])
  vertex_positions += np.array([i * 0.5 / NUM_LOOPS, 0, 0])
  # rotate all points by x axis
  vertex_positions = np.dot(vertex_positions, np.array([[1, 0, 0], [0, math.cos(np.pi * i / (NUM_LOOPS + 1)), -math.sin(np.pi * i / (NUM_LOOPS + 1))], [0, math.sin(np.pi * i / (NUM_LOOPS + 1)), math.cos(np.pi * i / (NUM_LOOPS + 1))]]))
  vertex_positions_value[i * NUM_POINTS: (i + 1) * NUM_POINTS, :] = vertex_positions
  # get the average length of the edges
  edge_length = np.linalg.norm(vertex_positions - np.roll(vertex_positions, shift=1, axis=0), axis=1).mean()
  print(f"Loop {i} has edge length {edge_length}")


# pre compute edge pair indices
same_loop_edge_pair_indices = []
different_loop_edge_pair_indices = []

for i in range(NUM_LOOPS):
  # first we add edge pair for edges within the same loop
  for j in range(NUM_POINTS):
    current_edge_start = i * NUM_POINTS + j
    currentt_edge_end = i * NUM_POINTS + (j + 1) % NUM_POINTS
    for k in range(j + 2, NUM_POINTS):
      next_edge_start = i * NUM_POINTS + k % NUM_POINTS
      next_edge_end = i * NUM_POINTS + (k + 1) % NUM_POINTS
      if next_edge_end == current_edge_start:
        break
      same_loop_edge_pair_indices.append([current_edge_start, currentt_edge_end, next_edge_start, next_edge_end])
  # now for edges between loops
  for j in range(NUM_POINTS):
    current_edge_start = i * NUM_POINTS + j
    currentt_edge_end = i * NUM_POINTS + (j + 1) % NUM_POINTS
    for k in range(i + 1, NUM_LOOPS):
      for m in range(NUM_POINTS):
        next_edge_start = k * NUM_POINTS + m
        next_edge_end = k * NUM_POINTS + (m + 1) % NUM_POINTS
        different_loop_edge_pair_indices.append([current_edge_start, currentt_edge_end, next_edge_start, next_edge_end])

print(f"There are {len(same_loop_edge_pair_indices)} same loop pairs")
print(f"There are {len(different_loop_edge_pair_indices)} different loop pairs")

s = scene("scene")
m = s.addMesh("mesh")
m.addAttribute("same_repulsive_weight", rows = 1, cols = 1)
m.attributes["same_repulsive_weight"].updateValue(np.array([50.0], dtype=np.float64))
m.addAttribute("diff_repulsive_weight", rows = 1, cols = 1)
m.attributes["diff_repulsive_weight"].updateValue(np.array([50.0], dtype=np.float64))
m.addAttribute("barrier_weight", rows = 1, cols = 1)
m.attributes["barrier_weight"].updateValue(np.array([1000.0], dtype=np.float64))
m.addAttribute("alpha", rows = 1, cols = 1)
m.attributes["alpha"].updateValue(np.array([2.0], dtype=np.float64))
m.addAttribute("beta", rows = 1, cols = 1)
m.attributes["beta"].updateValue(np.array([4.5], dtype=np.float64))

vertex = m.addPrimitive("vertex", NUM_POINTS * NUM_LOOPS) # add vertices
same_edge_pairs = m.addPrimitive("same_edge_pair", len(same_loop_edge_pair_indices)) # add edge pairs
diff_edge_pairs = m.addPrimitive("diff_edge_pair", len(different_loop_edge_pair_indices)) # add edge pairs

vertex_positions = vertex.addAttribute("rest_position", rows = 3, cols = 1)
vertex_positions.updateValue(vertex_positions_value)
# making it more complicated by adding translation and scale to the points
translations = vertex.addAttribute("translation", rows = 3, cols = 1)
translations.updateValue(np.zeros((NUM_POINTS * NUM_LOOPS, 3)))
scales = vertex.addAttribute("scale", rows = 1, cols = 1)
scales.updateValue(np.ones((NUM_POINTS * NUM_LOOPS, 1)))

# add the relation between edge pairs and vertices
c0 = same_edge_pairs.addConnectivity("edge_pair_to_vertex", vertex, same_loop_edge_pair_indices, 4)
c1 = diff_edge_pairs.addConnectivity("edge_pair_to_vertex", vertex, different_loop_edge_pair_indices, 4)

true_position = vertex_positions
same_edge_pair_positions = same_edge_pairs.addAttribute("position", through = c0, source = true_position)
diff_edge_pair_positions = diff_edge_pairs.addAttribute("position", through = c1, source = true_position)


# add the sphere boundary energy
sphere_boundary_energy = attribute.select((true_position).norm() > 1.0, 100.0 * ((true_position).norm() - 1.0), -(1.0 - true_position.norm()).log())
# sphere_boundary_energy = -(1.0 - true_position.norm()).log()
sphere_boundary_energy = vertex.addAttribute("sphere_boundary_energy", computed_attribute = sphere_boundary_energy * m.attributes["barrier_weight"])

# add the repulsive force
def repulsive_energy(points, alpha, beta):
  p0 = points.row(0)
  p1 = points.row(1)
  p2 = points.row(2)
  p3 = points.row(3)
  # r = p0.dot_explicit(p0) + p1.dot_explicit(p1) + p2.dot_explicit(p2) + p3.dot_explicit(p3)
  T01 = (p1 - p0) / ((p1 - p0).dot_explicit(p1 - p0)).sqrt()
  T23 = (p3 - p2) / ((p3 - p2).dot_explicit(p3 - p2)).sqrt()
  r = T01.dot_explicit(p0 - p2).pow(alpha) / (p0 - p2).norm().pow(beta)
  r += T01.dot_explicit(p0 - p3).pow(alpha) / (p0 - p3).norm().pow(beta)
  r += T01.dot_explicit(p1 - p2).pow(alpha) / (p1 - p2).norm().pow(beta)
  r += T01.dot_explicit(p1 - p3).pow(alpha) / (p1 - p3).norm().pow(beta)
  r += T23.dot_explicit(p2 - p0).pow(alpha) / (p2 - p0).norm().pow(beta)
  r += T23.dot_explicit(p2 - p1).pow(alpha) / (p2 - p1).norm().pow(beta)
  r += T23.dot_explicit(p3 - p0).pow(alpha) / (p3 - p0).norm().pow(beta)
  r += T23.dot_explicit(p3 - p1).pow(alpha) / (p3 - p1).norm().pow(beta)
  # make the segment longer
  r += 1.0 / (p0 - p1).dot(p0 - p1)
  return r

same_repulsive_energy_value = repulsive_energy(same_edge_pair_positions, m["alpha"], m["beta"])
same_repulsive_energy = same_edge_pairs.addAttribute("same_repulsive_energy", computed_attribute = same_repulsive_energy_value * m.attributes["same_repulsive_weight"])

diff_repulsive_energy_value = repulsive_energy(diff_edge_pair_positions, m["alpha"], m["beta"])
diff_repulsive_energy = diff_edge_pairs.addAttribute("diff_repulsive_energy", computed_attribute = diff_repulsive_energy_value * m.attributes["diff_repulsive_weight"])

s.addEnergy(same_repulsive_energy)
s.addEnergy(diff_repulsive_energy)
s.addEnergy(sphere_boundary_energy)
s.addMinimizeTarget([vertex_positions])

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.ion()
plt.axis('off')
plt.gca().set_aspect('equal', adjustable='box')
fig = plt.figure(figsize=(6, 8))
ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot
# Initialize the line object for updating
all_lines = []
for i in range(NUM_LOOPS):
  line,  = ax.plot([], [], [], linewidth=2)
  all_lines.append(line)
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
# for i in range(NUM_LOOPS):
#   x = [vertex_positions_value[j + i * NUM_POINTS, 0] for j in range(NUM_POINTS)]
#   y = [vertex_positions_value[j + i * NUM_POINTS, 1] for j in range(NUM_POINTS)]
#   z = [vertex_positions_value[j + i * NUM_POINTS, 2] for j in range(NUM_POINTS)]
#   all_lines[i].set_data(x, y)
#   all_lines[i].set_3d_properties(z)
# plt.show()

def plot_segments(points):
  # Extract coordinates
  for i in range(NUM_LOOPS):
    x = [points[j + i * NUM_POINTS, 0] for j in range(NUM_POINTS)]
    y = [points[j + i * NUM_POINTS, 1] for j in range(NUM_POINTS)]
    z = [points[j + i * NUM_POINTS, 2] for j in range(NUM_POINTS)]
    all_lines[i].set_data(x, y)
    all_lines[i].set_3d_properties(z)
  fig.canvas.draw()
  fig.canvas.flush_events()

weight = 0.05
for i in range(10000):
  result = s.minimizeEnergy()
  updated_value = (vertex_positions.value - weight * result[0]).get().reshape(NUM_POINTS * NUM_LOOPS, 3)
  # # vertex_positions.updateValue(vertex_positions.value - 0.01 * result[0])
  # Reshape updated_value to a 3D array:
  # Shape: (NUM_LOOPS, NUM_POINTS, 3)
  updated_value = updated_value.reshape(NUM_LOOPS, NUM_POINTS, 3)

  # Perform the vectorized computation within each loop
  updated_value = (
      0.8 * updated_value +
      0.1 * np.roll(updated_value, shift=-1, axis=1) +  # Next point in the same loop
      0.1 * np.roll(updated_value, shift=1, axis=1)     # Previous point in the same loop
  )
  # If needed, reshape updated_value back to its original shape
  updated_value = updated_value.reshape(-1, 3)
  vertex_positions.updateValue(updated_value)
  plot_segments(updated_value)
  if i % 1000 == 0:
    # save the updated value to a file using numpy
    np.save(f"output_{i}.npy", updated_value)
