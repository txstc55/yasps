from yasps.scene import scene
from yasps.attribute import attribute
import numpy as np
import math
np.random.seed(13)
NUM_POINTS = 5
def generate_points_near_equator(num_points, delta_phi_deg=5):
  theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
  delta_phi_rad = np.radians(delta_phi_deg)
  phi = np.random.uniform(-delta_phi_rad, delta_phi_rad, num_points)
  x = np.cos(phi) * np.cos(theta)
  y = np.cos(phi) * np.sin(theta)
  z = np.sin(phi)
  points = np.vstack((x, y, z)).T
  return points
vertex_positions_value = generate_points_near_equator(NUM_POINTS) * 1.0 # generate points inside the sphere
print(vertex_positions_value)


edge_pair_indices = []
for i in range(NUM_POINTS):
  current_edge_start = i
  current_edge_end = (i + 1) % NUM_POINTS
  for j in range(i + 2, NUM_POINTS):
    next_edge_start = j % NUM_POINTS
    next_edge_end = (j + 1) % NUM_POINTS
    if next_edge_end == current_edge_start:
      break
    edge_pair_indices.append([current_edge_start, current_edge_end, next_edge_start, next_edge_end])
print(f"There are {len(edge_pair_indices)} pairs")

s = scene("scene")
m = s.addMesh("mesh")
m.addAttribute("repulsive_weight", rows = 1, cols = 1)
m.attributes["repulsive_weight"].updateValue(np.array([1.0]))
m.addAttribute("barrier_weight", rows = 1, cols = 1)
m.attributes["barrier_weight"].updateValue(np.array([1.0]))
m.addAttribute("alpha", rows = 1, cols = 1)
m.attributes["alpha"].updateValue(np.array([2.0]))
m.addAttribute("beta", rows = 1, cols = 1)
m.attributes["beta"].updateValue(np.array([4.5]))

vertex = m.addPrimitive("vertex", NUM_POINTS) # add vertices
edge_pairs = m.addPrimitive("edge_pair", len(edge_pair_indices)) # add edge pairs

vertex_positions = vertex.addAttribute("position", rows = 3, cols = 1)
vertex_positions.updateValue(vertex_positions_value)

# add the relation between edge pairs and vertices
c = edge_pairs.addConnectivity("edge_pair_to_vertex", vertex, edge_pair_indices, 4)
edge_pair_positions = edge_pairs.addAttribute("position", through = c, source = vertex_positions)

# add the sphere boundary energy
sphere_boundary_energy = attribute.select(vertex_positions.norm() > 1.0, vertex_positions.norm() - 1.0, attribute(float_value = 0.0))
vertex.addAttribute("sphere_boundary_energy", computed_attribute = sphere_boundary_energy)


# add the repulsive force
def repulsive_energy(points, repulsive_weight, alpha, beta):
  p0 = points.row(0)
  p1 = points.row(1)
  p2 = points.row(2)
  p3 = points.row(3)
  T01 = (p1 - p0) / ((p1 - p0).dot_explicit(p1 - p0)).sqrt()
  T23 = (p3 - p2) / ((p3 - p2).dot_explicit(p3 - p2)).sqrt()
  # r = (p1) / 2.0
  r = T01.dot_explicit(p0 - p2).pow(alpha) / (p0 - p2).norm().pow(beta)
  r += T01.dot_explicit(p0 - p3).pow(alpha) / (p0 - p3).norm().pow(beta)
  r += T01.dot_explicit(p1 - p2).pow(alpha) / (p1 - p2).norm().pow(beta)
  r += T01.dot_explicit(p1 - p3).pow(alpha) / (p1 - p3).norm().pow(beta)
  r += T23.dot_explicit(p2 - p0).pow(alpha) / (p2 - p0).norm().pow(beta)
  r += T23.dot_explicit(p2 - p1).pow(alpha) / (p2 - p1).norm().pow(beta)
  r += T23.dot_explicit(p3 - p0).pow(alpha) / (p3 - p0).norm().pow(beta)
  r += T23.dot_explicit(p3 - p1).pow(alpha) / (p3 - p1).norm().pow(beta)
  # return r
  return r * repulsive_weight / 4.0

repulsive_energy_value = repulsive_energy(edge_pair_positions, m["repulsive_weight"], m["alpha"], m["beta"])
repulsive_energies = edge_pairs.addAttribute("repulsive_energy", computed_attribute = repulsive_energy_value)
print(repulsive_energies.compute().value.get())

# manually test out the energy
p0 = vertex_positions_value[0, :]
p1 = vertex_positions_value[1, :]
p2 = vertex_positions_value[2, :]
p3 = vertex_positions_value[3, :]
p4 = vertex_positions_value[4, :]
def repulsive_energy_np(p0, p1, p2, p3, repulsive_weight, alpha, beta):
  T01 = (p1 - p0) / math.sqrt((p1 - p0).dot(p1 - p0))
  T23 = (p3 - p2) / math.sqrt((p3 - p2).dot(p3 - p2))
  r = math.pow(T01.dot(p0 - p2), alpha) / math.pow(np.linalg.norm(p0 - p2), beta)
  r += math.pow(T01.dot(p0 - p3), alpha) / math.pow(np.linalg.norm(p0 - p3), beta)
  r += math.pow(T01.dot(p1 - p2), alpha) / math.pow(np.linalg.norm(p1 - p2), beta)
  r += math.pow(T01.dot(p1 - p3), alpha) / math.pow(np.linalg.norm(p1 - p3), beta)
  r += math.pow(T23.dot(p2 - p0), alpha) / math.pow(np.linalg.norm(p2 - p0), beta)
  r += math.pow(T23.dot(p2 - p1), alpha) / math.pow(np.linalg.norm(p2 - p1), beta)
  r += math.pow(T23.dot(p3 - p0), alpha) / math.pow(np.linalg.norm(p3 - p0), beta)
  r += math.pow(T23.dot(p3 - p1), alpha) / math.pow(np.linalg.norm(p3 - p1), beta)
  return r * repulsive_weight / 4.0


# print(repulsive_energy_np(p0, p1, p2, p3, 1.0, 2.0, 4.5))
# print(repulsive_energy_np(p0, p1, p3, p4, 1.0, 2.0, 4.5))
#
from yasps import autodiff
ad = autodiff()
energy_grad = ad.diff(repulsive_energies, edge_pair_positions)
edge_pairs.addAttribute("grad", computed_attribute = energy_grad)
print(energy_grad.compute().value.get())
