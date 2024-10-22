import numpy as np
import matplotlib.pyplot as plt
np.random.seed(13)
# Data points
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
points = generate_points_near_equator(NUM_POINTS) * 1.0 # generate points inside the sphere

# Extract coordinates
x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# # Plot the points as a scatter plot
# ax.scatter(x[:-1], y[:-1], z[:-1], c='blue', marker='o')  # Exclude the last point in scatter

# Plot lines connecting the points to form a loop
ax.plot(x, y, z, c='red', linewidth=2)

# Label axes
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

# Set title
ax.set_title('3D Loop Plot of Provided Points')

# Show plot
plt.show()
