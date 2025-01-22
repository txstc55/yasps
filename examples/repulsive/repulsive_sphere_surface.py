from yasps import scene
from yasps.attribute import attribute
import numpy as np
np.random.seed(13)
NUM_POINTS = 1000
NUM_LOOPS = 2
def generate_points_near_equator(num_points, delta_phi_deg=5):
  theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False, dtype=np.float64)
  # delta_phi_rad = np.radians(delta_phi_deg)
  # phi = np.zeros(num_points)
  phi = np.random.uniform(-0.01, 0.01, num_points)
  x = np.cos(phi) * np.cos(theta)
  y = np.cos(phi) * np.sin(theta)
  z = np.sin(phi)
  # normalize
  norm = np.linalg.norm(np.vstack((x, y, z)).T, axis=1)
  points = np.vstack((x, y, z)).T / norm[:, None]
  return points

vertex_positions_value = np.empty((NUM_POINTS * NUM_LOOPS, 3), dtype=np.float64)

vertex_positions0 = generate_points_near_equator(NUM_POINTS)
# normalize the points so they are on the sphere
vertex_positions0 = vertex_positions0 / np.linalg.norm(vertex_positions0, axis=1)[:, None]
# translate each point by z axis
vertex_positions0 = vertex_positions0 + np.array([0., 0.0, 0.5])
# normalize the points so they are on the sphere
vertex_positions0 = vertex_positions0 / np.linalg.norm(vertex_positions0, axis=1)[:, None]
vertex_positions_value[0: NUM_POINTS, :] = vertex_positions0


vertex_positions1 = generate_points_near_equator(NUM_POINTS)
vertex_positions1 = vertex_positions1 / np.linalg.norm(vertex_positions1, axis=1)[:, None]
vertex_positions1 = vertex_positions1 + np.array([0., 0, 0.2])
vertex_positions1 = vertex_positions1 / np.linalg.norm(vertex_positions1, axis=1)[:, None]
vertex_positions_value[NUM_POINTS: 2 * NUM_POINTS, :] = vertex_positions1

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
    for k in range(NUM_LOOPS):
      if k != i:
        for m in range(NUM_POINTS):
          next_edge_start = k * NUM_POINTS + m
          next_edge_end = k * NUM_POINTS + (m + 1) % NUM_POINTS
          different_loop_edge_pair_indices.append([current_edge_start, currentt_edge_end, next_edge_start, next_edge_end])

print(f"There are {len(same_loop_edge_pair_indices)} same loop pairs")
print(f"There are {len(different_loop_edge_pair_indices)} different loop pairs")

s = scene("scene")
m = s.addMesh("mesh")
m.addAttribute("same_repulsive_weight", rows = 1, cols = 1)
m.attributes["same_repulsive_weight"].updateValue(np.array([1.0], dtype=np.float64))
m.addAttribute("diff_repulsive_weight", rows = 1, cols = 1)
m.attributes["diff_repulsive_weight"].updateValue(np.array([1.0], dtype=np.float64))
m.addAttribute("alpha", rows = 1, cols = 1)
m.attributes["alpha"].updateValue(np.array([3.0], dtype=np.float64))
m.addAttribute("beta", rows = 1, cols = 1)
m.attributes["beta"].updateValue(np.array([6.0], dtype=np.float64))

vertex = m.addPrimitive("vertex", NUM_POINTS * NUM_LOOPS) # add vertices
same_edge_pairs = m.addPrimitive("same_edge_pair", len(same_loop_edge_pair_indices)) # add edge pairs
diff_edge_pairs = m.addPrimitive("diff_edge_pair", len(different_loop_edge_pair_indices)) # add edge pairs

vertex_positions = vertex.addAttribute("position", rows = 3, cols = 1)
vertex_positions.updateValue(vertex_positions_value)

# add the relation between edge pairs and vertices
c0 = same_edge_pairs.addConnectivity("edge_pair_to_vertex", vertex, same_loop_edge_pair_indices, 4)
c1 = diff_edge_pairs.addConnectivity("edge_pair_to_vertex", vertex, different_loop_edge_pair_indices, 4)

same_edge_pair_positions = same_edge_pairs.addAttribute("position", through = c0, source = vertex_positions)
diff_edge_pair_positions = diff_edge_pairs.addAttribute("position", through = c1, source = vertex_positions)


# add the repulsive force
def repulsive_energy_same_loop(points, alpha, beta):
  p0 = points.row(0)
  p1 = points.row(1)
  p2 = points.row(2)
  p3 = points.row(3)
  p0 = p0 / p0.norm()
  p1 = p1 / p1.norm()
  p2 = p2 / p2.norm()
  p3 = p3 / p3.norm()
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

# add the repulsive force
def repulsive_energy_diff_loop(points, alpha, beta):
  p0 = points.row(0)
  p1 = points.row(1)
  p2 = points.row(2)
  p3 = points.row(3)
  p0 = p0 / p0.norm()
  p1 = p1 / p1.norm()
  p2 = p2 / p2.norm()
  p3 = p3 / p3.norm()
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
  return r

same_repulsive_energy_value = repulsive_energy_same_loop(same_edge_pair_positions, m["alpha"], m["beta"])
same_repulsive_energy = same_edge_pairs.addAttribute("same_repulsive_energy", computed_attribute = same_repulsive_energy_value * m.attributes["same_repulsive_weight"])

diff_repulsive_energy_value = repulsive_energy_same_loop(diff_edge_pair_positions, m["alpha"], m["beta"])
diff_repulsive_energy = diff_edge_pairs.addAttribute("diff_repulsive_energy", computed_attribute = diff_repulsive_energy_value * m.attributes["diff_repulsive_weight"])

# # we will also two barrier energies to make it repulsive from the shell
# # the first shell is with radius 0.9, the second is 1.1
# # first we get the point's norm
# BARRIER_DISTANCE = 0.4
# point_norm = vertex_positions.norm()
# point_normalized = vertex_positions / point_norm
# # get the projected point on the sphere
# projected_point0 = point_normalized * 0.98
# projected_point1 = point_normalized * 1.02
# distance0 = (projected_point0 - vertex_positions).norm()
# distance1 = (projected_point1 - vertex_positions).norm()
# barrier_energy0 = attribute.select(distance0 > BARRIER_DISTANCE, attribute(float_value = 0.0), -(distance0 - BARRIER_DISTANCE) * (distance0 - BARRIER_DISTANCE) * (distance0 / BARRIER_DISTANCE).log())
# barrier_energy1 = attribute.select(distance1 > BARRIER_DISTANCE, attribute(float_value = 0.0), -(distance1 - BARRIER_DISTANCE) * (distance1 - BARRIER_DISTANCE) * (distance1 / BARRIER_DISTANCE).log())
# barrier_energy = vertex.addAttribute("barrier_energy", computed_attribute =  barrier_energy0 + barrier_energy1)



s.addEnergy(same_repulsive_energy)
s.addEnergy(diff_repulsive_energy)
# s.addEnergy(barrier_energy)
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
plt.show()

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

# plot_segments(vertex_positions_value)
# plt.show()

weight = 0.05
for i in range(500):
  result = s.minimizeEnergy()
  result = result[0].get().reshape(-1, 3)

  updated_value = (vertex_positions.value.get().reshape(-1, 3) - weight * result)
  # normalize it
  updated_value = updated_value / np.linalg.norm(updated_value, axis=1)[:, None]
  # get the maximum change's norm
  updated_value = updated_value.reshape(NUM_LOOPS, NUM_POINTS, 3)

  # # Perform the vectorized computation within each loop
  updated_value = (
      0.9 * updated_value +
      0.05 * np.roll(updated_value, shift=-1, axis=1) +  # Next point in the same loop
      0.05 * np.roll(updated_value, shift=1, axis=1)     # Previous point in the same loop
  )
  # If needed, reshape updated_value back to its original shape
  updated_value = updated_value.reshape(-1, 3)
  vertex_positions.updateValue(updated_value)
  plot_segments(updated_value)
  if i % 1 == 0:
    # save the updated value to a file using numpy
    np.save(f"repulsive_data/output_surface_{i}_config1.npy", updated_value)
