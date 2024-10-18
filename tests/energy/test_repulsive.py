from yasps.scene import scene
import numpy as np
NUM_POINTS = 1003
# generate num_points on a sphere on the equitor
def generate_points_on_equator(num_points):
  # Generate an array of evenly spaced angles from 0 to 2π
  angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
  # Calculate x and y coordinates for each angle, with z = 0
  x = np.cos(angles)
  y = np.sin(angles)
  z = np.zeros(num_points)  # z-coordinates are all zero on the equator
  # Combine x, y, z coordinates into a 2D array of shape (num_points, 3)
  points = np.vstack((x, y, z)).T
  return points

vertex_positions = generate_points_on_equator(NUM_POINTS) * 0.8




s = scene("scene")
m = s.addMesh("mesh")
m.addPrimitive("vertex", NUM_POINTS) # add 1002 vertices
m.addPrimitive("edge_pair", (NUM_POINTS - 3) * (NUM_POINTS - 3)) # each edge pairs with 1000 other edges since it cannot connect to itself and neighboring 2 edges
