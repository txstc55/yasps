from yasps import scene
from yasps import attribute
import math
import numpy as np

def quaternion_to_rotation(w, z):
  # quaternion to a 2d rotation matrix
  return attribute.to_array([w * w - z * z, -2 * w * z, 2 * w * z, w * w - z * z], rows = 2, cols = 2)

# define segments
SEGMENT_COUNT = 5
TARGET_POSITION = [4.0, 0.0]
OBSTACLE_POSITION0 = [-0.8, 0.0]
OBSTACLE_POSITION1 = [0.8, 0.0]
OBSTACLE_POSITION0_ATTRIBUTE = attribute.to_array([attribute(float_value = OBSTACLE_POSITION0[0]), attribute(float_value = OBSTACLE_POSITION0[1])], rows = 2, cols = 1)
OBSTACLE_POSITION1_ATTRIBUTE = attribute.to_array([attribute(float_value = OBSTACLE_POSITION1[0]), attribute(float_value = OBSTACLE_POSITION1[1])], rows = 2, cols = 1)
s0 = scene("scene0")
pendulum = s0.addMesh("pendulum")
segments = []
for i in range(SEGMENT_COUNT):
  segment = pendulum.addPrimitive(f"segment{i}", numInstances = 1)
  # local quaternion
  segment.addAttribute("w", rows = 1, cols = 1)
  segment.addAttribute("z", rows = 1, cols = 1)
  segment.addAttribute("length", rows = 1, cols = 1)
  segment["w"].updateValue([math.cos(0)])
  segment["z"].updateValue([math.sin(0)])
  segment["length"].updateValue([1.0])
  segments.append(segment)

segments[0]["w"].updateValue([math.cos(math.pi / 8)])
segments[0]["z"].updateValue([math.sin(math.pi / 8)])
segments[1]["w"].updateValue([math.cos(-math.pi / 8)])
segments[1]["z"].updateValue([math.sin(-math.pi / 8)])
segments[2]["w"].updateValue([math.cos(-math.pi / 6)])
segments[2]["z"].updateValue([math.sin(-math.pi / 6)])

# the rest position of segment 0
segments[0].addAttribute("global_rotation", computed_attribute = quaternion_to_rotation(segments[0]["w"], segments[0]["z"]))
segments[0].addAttribute("previous_end_position", rows = 2, cols = 1)
segments[0]["previous_end_position"].updateValue([0.0, 0.0])
segments[0].addAttribute("end_point_position", computed_attribute = segments[0]["global_rotation"] * attribute.to_array([attribute(float_value = 0.0), -segments[0]["length"]], rows = 2, cols = 1))

# first we add the connectivity to the last element
for i in range(1, SEGMENT_COUNT):
  connect = segments[i].addConnectivity("previous", segments[i - 1], [0], 1)
  segments[i].addAttribute("local_rotation", computed_attribute = quaternion_to_rotation(segments[i]["w"], segments[i]["z"]))
  segments[i].addAttribute("previous_global_rotation", through = connect, source = segments[i - 1]["global_rotation"])
  segments[i]["previous_global_rotation"].reshape(2, 2)
  segments[i].addAttribute("previous_end_position", through = connect, source = segments[i - 1]["end_point_position"])
  segments[i]["previous_end_position"].reshape(2, 1)
  segments[i].addAttribute("global_rotation", computed_attribute = segments[i]["previous_global_rotation"] * segments[i]["local_rotation"])
  segments[i].addAttribute("end_point_position", computed_attribute = segments[i]["global_rotation"] * attribute.to_array([attribute(float_value = 0.0), -segments[i]["length"]], rows = 2, cols = 1) + segments[i]["previous_end_position"])


# for convenience, we add a primitive to store the vertices
vertices = pendulum.addPrimitive("vertices", numInstances = SEGMENT_COUNT + 1)
weighted_position = attribute.to_array([attribute(float_value = 0.0), attribute(float_value = 0.0)], rows = 2, cols = 1)
for i in range(SEGMENT_COUNT):
  weights = [0.0 for _ in range(1 + SEGMENT_COUNT)]
  weights[i + 1] = 1.0
  w = vertices.addAttribute(f"w_{i}", rows = 1, cols = 1)
  w.updateValue(weights)
  conn = vertices.addConnectivity(f"to_segment_{i}", segments[i], [0] * (SEGMENT_COUNT + 1), 1)
  epp = vertices.addAttribute(f"epp{i}", through = conn, source = segments[i]["end_point_position"])
  epp.reshape(2, 1)
  weighted_position = weighted_position.add_explicit(w.mul_explicit(epp))
vertex_positions = vertices.addAttribute("position", computed_attribute = weighted_position)

segments[-1].addAttribute("target_position", rows = 2, cols = 1)
segments[-1]["target_position"].updateValue(TARGET_POSITION)
distance_vector = segments[-1]["end_point_position"] - segments[-1]["target_position"]
segments[-1].addAttribute("position_penalty", computed_attribute = (segments[-1]["end_point_position"] - segments[-1]["target_position"]).norm().pow(2.0))


s0.addEnergy(segments[-1]["position_penalty"])

def repulsive_energy(p0, p1, o, alpha, beta):
  SE = p1 - p0
  SE_norm = SE.norm()
  T = SE / SE_norm  # Unit tangent vector
  v0 = p0 - o
  v1 = p1 - o
  T_dot_v0 = T.dot(v0)
  T_dot_v1 = T.dot(v1)
  norm_v0 = v0.norm()
  norm_v1 = v1.norm()
  epsilon = 1e-8
  norm_v0 = norm_v0 + epsilon
  norm_v1 = norm_v1 + epsilon
  energy_p0 = (T_dot_v0.pow(alpha)) / (norm_v0.pow(beta))
  energy_p1 = (T_dot_v1.pow(alpha)) / (norm_v1.pow(beta))
  r = (energy_p0 + energy_p1)
  return r

for i in range(SEGMENT_COUNT):
  segments[i].addAttribute("repulsive", computed_attribute = repulsive_energy(segments[i]["previous_end_position"], segments[i]["end_point_position"], OBSTACLE_POSITION0_ATTRIBUTE, 3.0, 6.0) + repulsive_energy(segments[i]["previous_end_position"], segments[i]["end_point_position"], OBSTACLE_POSITION1_ATTRIBUTE, 3.0, 6.0))
  s0.addEnergy(segments[i]["repulsive"])


s0.addMinimizeTarget([segment["w"] for segment in segments] + [segment["z"] for segment in segments])
result = s0.minimizeEnergy()

import matplotlib.pyplot as plt
data = vertex_positions.compute().value.get()
points = data.reshape(-1, 2)
x = points[:, 0]
y = points[:, 1]
plt.ion()
fig, ax = plt.subplots(figsize=(6, 8))
line, = ax.plot([], [], marker='o', linestyle='-')
line.set_data(x, y)
ax.set_xlim(-(SEGMENT_COUNT + 2), (SEGMENT_COUNT + 2))
ax.set_ylim(-(SEGMENT_COUNT + 2), (SEGMENT_COUNT + 2))
ax.plot(TARGET_POSITION[0], TARGET_POSITION[1], 'ro', color='cyan', label = 'Target Position')
ax.plot(OBSTACLE_POSITION0[0], OBSTACLE_POSITION0[1], 'ro', color='red', label = 'Target Position')
ax.plot(OBSTACLE_POSITION1[0], OBSTACLE_POSITION1[1], 'ro', color='red', label = 'Target Position')
# ax.legend()
plt.axis('off')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# Loop to update the plot
t = 0.1
last_penalty = 50
for iteration in range(1000):
  result = s0.minimizeEnergy()
  dx = (np.array([x.get() for x in result]).flatten())
  print("Penalty value")
  last_penalty = segments[-1]["position_penalty"].compute().value.get()[0]
  print(last_penalty)

  for i in range(SEGMENT_COUNT):
    w_origin = segments[i]["w"].compute().value.get()
    z_origin = segments[i]["z"].compute().value.get()
    d_w = dx[i]
    d_z = dx[i + SEGMENT_COUNT]
    w_new = w_origin - t * d_w
    z_new = z_origin - t * d_z
    norm = math.sqrt(w_new * w_new + z_new * z_new)
    w_new /= norm
    z_new /= norm
    segments[i]["w"].updateValue([w_new])
    segments[i]["z"].updateValue([z_new])

  data = vertex_positions.compute().value.get()
  points = data.reshape(-1, 2)
  x = points[:, 0]
  y = points[:, 1]
  line.set_data(x, y)

  # Save the current frame as an image
  plt.savefig(f'pendulum_results/frame_{iteration:04d}.png', dpi=600)
  fig.canvas.draw()
  fig.canvas.flush_events()
  # 1890, 1257, 1004, 2055 use this paramater to crop image

# Disable interactive mode if no longer needed
plt.ioff()
plt.show()
