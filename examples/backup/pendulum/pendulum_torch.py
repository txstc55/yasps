import torch
import math

# Define segment lengths
SEGMENT_COUNT = 5
segment_lengths = [1.0 for i in range(SEGMENT_COUNT)]


# Define the target point
TARGET_POSITION = torch.tensor([4.0, 0.0])
OBSTACLE_POSITION0 = torch.tensor([-0.8, 0.0])
OBSTACLE_POSITION1 = torch.tensor([0.8, 0.0])

# Initialize segments with their lengths and quaternions
segments = [{} for _ in range(5)]

# Set lengths
for i in range(5):
  segments[i]['length'] = segment_lengths[i]

# Set fixed quaternions for segments 0-2
segments[0]['w'] = torch.tensor([math.cos(math.pi / 8)], requires_grad=True)
segments[0]['z'] = torch.tensor([math.sin(math.pi / 8)], requires_grad=True)
segments[1]['w'] = torch.tensor([math.cos(-math.pi / 8)], requires_grad=True)
segments[1]['z'] = torch.tensor([math.sin(-math.pi / 8)], requires_grad=True)
segments[2]['w'] = torch.tensor([math.cos(-math.pi / 6)], requires_grad=True)
segments[2]['z'] = torch.tensor([math.sin(-math.pi / 6)], requires_grad=True)

# Initialize quaternions for segments 3 and 4 (variables to optimize)
initial_angle = 0.0  # Initial guess for the angle
segments[3]['w'] = torch.tensor([math.cos(initial_angle)], requires_grad=True)
segments[3]['z'] = torch.tensor([math.sin(initial_angle)], requires_grad=True)

segments[4]['w'] = torch.tensor([math.cos(initial_angle)], requires_grad=True)
segments[4]['z'] = torch.tensor([math.sin(initial_angle)], requires_grad=True)

def forward_kinematics(segments):
  """
  Compute the positions of each joint given the quaternions.
  """
  positions = []
  current_pos = torch.tensor([0.0, 0.0])
  positions.append(current_pos.clone())
  cumulative_rotation = torch.eye(2)
  for segment in segments:
    w = segment['w']
    z = segment['z']
    # Normalize the quaternion
    norm = torch.sqrt(w**2 + z**2)
    w = w / norm
    z = z / norm
    # Compute rotation matrix components
    cos_theta = (w**2 - z**2).squeeze()
    sin_theta = (2 * w * z).squeeze()
    # Construct the rotation matrix
    R = torch.stack([
      torch.stack([cos_theta, -sin_theta]),
      torch.stack([sin_theta, cos_theta])
    ])
    # Ensure R is of shape (2, 2)
    R = R.reshape(2, 2)
    # Update cumulative rotation
    cumulative_rotation = torch.matmul(cumulative_rotation, R)
    # Compute end position of the segment
    length = segment['length']
    delta_pos = torch.matmul(cumulative_rotation, torch.tensor([0, -length]))
    current_pos = current_pos + delta_pos
    positions.append(current_pos.clone())
  return positions

def repulsive_energy(p0, p1, o, alpha, beta):
  # Compute the unit tangent vector T
  SE = p1 - p0
  SE_norm = torch.norm(SE)
  T = SE / SE_norm  # Unit tangent vector
  # Compute vectors from p0 and p1 to the obstacle
  v0 = p0 - o
  v1 = p1 - o
  # Compute dot products T . v0 and T . v1
  T_dot_v0 = torch.dot(T, v0)
  T_dot_v1 = torch.dot(T, v1)
  # Compute norms ||v0|| and ||v1||
  norm_v0 = torch.norm(v0)
  norm_v1 = torch.norm(v1)
  # Avoid division by zero by adding a small epsilon
  epsilon = 1e-8
  norm_v0 = norm_v0 + epsilon
  norm_v1 = norm_v1 + epsilon
  # Compute the repulsive energy components
  energy_p0 = (T_dot_v0 ** alpha) / (norm_v0 ** beta)
  energy_p1 = (T_dot_v1 ** alpha) / (norm_v1 ** beta)
  # Total repulsive energy
  r = (energy_p0 + energy_p1)
  return r


def loss_function(segments):
  """
  Compute the loss as the squared distance between the end effector and the target point.
  """
  positions = forward_kinematics(segments)
  end_effector = positions[-1]
  loss = (end_effector - TARGET_POSITION).norm()
  for i in range(SEGMENT_COUNT):
    p0 = positions[i]
    p1 = positions[i + 1]
    loss = loss + repulsive_energy(p0, p1, OBSTACLE_POSITION0, 3.0, 6.)
    loss = loss + repulsive_energy(p0, p1, OBSTACLE_POSITION1, 3.0, 6.)

  return loss

# Collect the parameters to optimize
parameters = [segments[0]['w'], segments[0]['z'], segments[1]['w'], segments[1]['z'], segments[2]['w'], segments[2]['z'], segments[3]['w'], segments[3]['z'], segments[4]['w'], segments[4]['z']]

# Set up the optimizer
optimizer = torch.optim.Adam(parameters, lr= 0.1)


import matplotlib.pyplot as plt
import numpy as np
points = np.array([item.detach() for item in forward_kinematics(segments)])
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


# Optimization loop
num_iterations = 1000
for i in range(num_iterations):
  optimizer.zero_grad()
  loss = loss_function(segments)
  loss.backward()
  optimizer.step()
  # Normalize the quaternions to ensure they remain unit quaternions
  for idx in range(len(segment_lengths)):
    w = segments[idx]['w']
    z = segments[idx]['z']
    norm = torch.sqrt(w.detach()**2 + z.detach()**2)
    segments[idx]['w'].data = w.data / norm
    segments[idx]['z'].data = z.data / norm
  if i % 1 == 0:
    print(f"Iteration {i}, Loss: {loss.item()}")
    points = np.array([item.detach() for item in forward_kinematics(segments)])
    x = points[:, 0]
    y = points[:, 1]
    line.set_data(x, y)
    fig.canvas.draw()
    fig.canvas.flush_events()
    # plt.savefig(f'pendulum_pytorch_results/frame_{i:04d}.png', dpi=600)

# After optimization, print the results
positions = forward_kinematics(segments)
end_effector = positions[-1]
print("\nOptimized Quaternions and Angles for Segments 3 and 4:")
for idx in range(len(segment_lengths)):
  w = segments[idx]['w'].detach()
  z = segments[idx]['z'].detach()
  theta = 2 * torch.atan2(z, w)
  angle_degrees = theta.item() * 180 / math.pi
  print(f"Segment {idx + 1}:")
  print(f"  Quaternion w: {w.item()}, z: {z.item()}")
  print(f"  Angle (degrees): {angle_degrees}")

print("\nFinal End Effector Position:", end_effector.detach().numpy())
print("Distance to Target Point:", torch.sqrt(loss_function(segments)).item())
