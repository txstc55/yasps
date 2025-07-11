from yasps import scene
from yasps import attribute
# import math
import numpy as np
from helpers import triangle_sphere_collision, inertia, triangle_center_distance_energy, tangent_energy, sigmoid
from yasps import minimizer
import faulthandler
import random
faulthandler.enable(all_threads=True)
# define PI
PI = 3.141592653589793
s0 = scene("scene0")
m = s0.addMesh("mesh")
DT_VALUE = 0.01
TANGENT_ENERGY_MODIFIER = 50.0
EDGE_ENERGY_MODIFIER = 1.0
WALL_ENERGY_MODIFIER = 1.0
HEIGHT_ENERGY_MODIFIER = 1.0
CENTER_DISTANCE_ENERGY_MODIFIER = 0.1


DT = s0.addAttribute("DT", rows = 1, cols = 1)
DT.updateValue([DT_VALUE]) # time step for the simulation

tl = m.addAttribute("triangle_length", rows = 1, cols = 1)
tl.updateValue([10.0])
bl = m.addAttribute("base_length", computed_attribute = tl * 1.5)

# now we add the base
arms = m.addPrimitive("arms", numInstances = 3)
base_angle_from_center = arms.addAttribute("base_angle_from_center", rows = 1, cols = 1)
base_angle_from_center.updateValue([PI / 6.0, 5.0 * PI /6.0, 9.0 * PI / 6.0])

base_position = arms.addAttribute("base_position", computed_attribute = attribute.to_array([bl * base_angle_from_center.cos(), 0.0, bl * base_angle_from_center.sin()], rows = 3, cols = 1)) # this is where the center of the base is

base_angle = arms.addAttribute("base_angle", rows = 1, cols = 1) # we allow the base to rotate around its center on the x-z plane
base_angle.updateValue([0, 0, 0])
base_height = arms.addAttribute("base_height", rows = 1, cols = 1) # the height of the base
base_height.updateValue([3.0, 3.0, 3.0]) # the height of the base

first_segment_length = arms.addAttribute("first_segment_length", rows = 1, cols = 1) # the length of the first segment
first_segment_length.updateValue([8.0, 8.0, 8.0])

first_segment_angle_sigmoid = arms.addAttribute("first_segment_angle_sigmoid", rows = 1, cols = 1) # the angle of the first segment
first_segment_angle_sigmoid.updateValue([0.0, 0.0, 0.0]) # the angle of the first segment, we allow it to rotate around z axis
first_segment_angle = arms.addAttribute("first_segment_angle", computed_attribute = PI / 4.0 + (sigmoid(first_segment_angle_sigmoid) * 2.0 - 1.0) * PI / 4.0)

# now we define the first segment
# since we only rotate the base around the x-z plane, the start point of the first segment is just the base position plus the height of the base
# so we just define the end position of the first segment
# we allow the first segment to rotate around z axis
first_segment_local_y = first_segment_angle.sin() * first_segment_length
first_segment_local_x = first_segment_angle.cos() * first_segment_length

# now we put it in the local frame of the base
first_segment_local_x_in_base = base_angle.cos() * first_segment_local_x
first_segment_local_z_in_base = base_angle.sin() * first_segment_local_x


# now we put it in the global frame without rotating the base around the center
first_segment_local_x_in_base = first_segment_local_x_in_base + bl

# now we have the x y z, we rotate it
first_segment_global_x = first_segment_local_x_in_base * base_angle_from_center.cos() + first_segment_local_z_in_base * base_angle_from_center.sin()
first_segment_global_y = first_segment_local_y + base_height
first_segment_global_z = first_segment_local_x_in_base * base_angle_from_center.sin() - first_segment_local_z_in_base * base_angle_from_center.cos()


# add the attribute
first_segment_end_position = arms.addAttribute("first_segment_end_position", computed_attribute=attribute.to_array([first_segment_global_x, first_segment_global_y, first_segment_global_z], rows=3, cols=1))

# now let's define the second segment
second_segment_length = arms.addAttribute("second_segment_length", rows = 1, cols = 1) # the length of the second segment
second_segment_length.updateValue([10.0, 10.0, 10.0])
second_segment_angle_sigmoid = arms.addAttribute("second_segment_angle_sigmoid", rows = 1, cols = 1) # the angle of the second segment
second_segment_angle_sigmoid.updateValue([0.0, 0.0, 0.0]) # the angle of the first segment, we allow it to rotate around the z axis
second_segment_angle = arms.addAttribute("second_segment_angle", computed_attribute = PI / 4.0 + (sigmoid(second_segment_angle_sigmoid) * 2.0 - 1.0) * PI / 4.0)

second_segment_local_y = second_segment_angle.sin() * second_segment_length + first_segment_local_y
second_segment_local_x = -second_segment_angle.cos() * second_segment_length + first_segment_local_x

second_segment_local_x_in_base = base_angle.cos() * second_segment_local_x
second_segment_local_z_in_base = base_angle.sin() * second_segment_local_x
second_segment_local_x_in_base = second_segment_local_x_in_base + bl

second_segment_global_x = second_segment_local_x_in_base * base_angle_from_center.cos() + second_segment_local_z_in_base * base_angle_from_center.sin()
second_segment_global_y = second_segment_local_y + base_height
second_segment_global_z = second_segment_local_x_in_base * base_angle_from_center.sin() - second_segment_local_z_in_base * base_angle_from_center.cos()

# add the attribute
second_segment_end_position = arms.addAttribute("second_segment_end_position", computed_attribute=attribute.to_array([second_segment_global_x, second_segment_global_y, second_segment_global_z], rows=3, cols=1))

# we also add a target height for the second segment
target_height = arms.addAttribute("target_height", rows = 1, cols = 1)
target_height.updateValue([10.0, 10.0, 10.0]) # we want the second segment to be at a height of 10 units

# add height penalty as energy
height_penalty = arms.addAttribute("height_penalty", computed_attribute = (second_segment_end_position.row(1) - target_height) * (second_segment_end_position.row(1) - target_height))


# we now make the primitive for the triangle
triangle = m.addPrimitive("triangle", numInstances = 1)
t2a = triangle.addConnectivity("triangle2arms", arms, [[0, 1, 2]], 3)
triangle_positions = triangle.addAttribute("triangle_positions", through = t2a, source = second_segment_end_position)

# make the triangle edges
triangle_edges = m.addPrimitive("triangle_edges", numInstances = 3)
e2a = triangle_edges.addConnectivity("edge2arms", arms, [[0, 1], [1, 2], [2, 0]], 2)
edge_positions = triangle_edges.addAttribute("edge_positions", through = e2a, source = second_segment_end_position)
edge_length = (edge_positions.row(0) - edge_positions.row(1)).norm()

edge_target_length = triangle_edges.addAttribute("edge_target_length", rows = 1, cols = 1)
edge_target_length.updateValue([30.0, 30.0, 30.0]) # we want the edges to be 25 units long
edge_energy_modifier = s0.addAttribute("edge_energy_modifier", rows = 1, cols = 1)
edge_energy_modifier.updateValue([EDGE_ENERGY_MODIFIER]) # this is the modifier for the edge energy
edge_energy = triangle_edges.addAttribute("edge_energy", computed_attribute = edge_energy_modifier * (edge_length - edge_target_length) * (edge_length - edge_target_length))

# we now make the pinballs
pinballs = m.addPrimitive("pinballs", numInstances = 1)
radius = pinballs.addAttribute("radius", rows = 1, cols = 1)
radius.updateValue([1, 0.5, 0.3, 1.0, 0.8])
pinball_positions = pinballs.addAttribute("pinball_positions", rows = 1, cols = 3)
# pinball_positions.updateValue([[0.0, 28.0, 0.0], [2.0, 37.0, 4.0], [3.0, 36.0, -2.0], [-2.0, 27.0, -3.0], [1.5, 29.0, -2.0]])
pinball_positions.updateValue([[4.0, 28.0, 3.0], [-2.0, 25.0, 4.0]][:1])
pinball_last_positions = pinballs.addAttribute("pinball_last_positions", rows = 1, cols = 3)
# pinball_last_positions.updateValue([[0.0, 28.0, 0.0], [2.0, 37.0, 4.0], [3.0, 36.0, -2.0], [-5.0, 27.0, -6.0], [3.0, 29.0, -4.0]])
pinball_last_positions.updateValue([[4.0, 28.0, 3.0], [2.0, 25.0, 4.0]][:1])

pinball_velocities = pinballs.addAttribute("pinball_velocities", rows = 1, cols = 3)
# pinball_velocities.updateValue([[-0.3, 5.0, 0.2], [1, 0.0, 0.5], [0.2, 0.0, 0], [0.0, 0.0, -0.1], [0.0, 0.0, 0.0]])
pinball_velocities.updateValue([[-0.8, 1.0, 1.5], [1, 0.0, -0.5]][:1])

pinball_masses = pinballs.addAttribute("pinball_masses", rows = 1, cols = 1)
pinball_masses.updateValue([100.0, 150.0, 1.0, 1.0, 1.0]) # we set the masses of the pinballs

p2a = pinballs.addConnectivity("pinball2arms", arms, [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]], 3)
pinball_triangle_positions = pinballs.addAttribute("pinball_triangle_positions", through = p2a, source = second_segment_end_position) # now this will be 3 by 3 indicating the 3 positions of the triangle for each pinball

KAPPA = s0.addAttribute("KAPPA", rows = 1, cols = 1)
KAPPA.updateValue([10.0]) # this is the spring constant for the pinballs
DELTA = s0.addAttribute("DELTA", rows = 1, cols = 1)
DELTA.updateValue([0.5]) # this is the delta for the pinballs
pinball_collide_triangle = triangle_sphere_collision(pinball_triangle_positions.row(0), pinball_triangle_positions.row(1), pinball_triangle_positions.row(2), pinball_positions, radius, DELTA, KAPPA)
# now we can add the collision energy for the pinballs
pinballs.addAttribute("pinball_collide_triangle", computed_attribute = pinball_collide_triangle)
inertia = inertia(pinball_last_positions, pinball_velocities, DT, pinball_positions, pinball_masses)
pinballs.addAttribute("inertia", computed_attribute = inertia)

# now we add energy so that the triangle try to move the pinballs to the center of the triangle
# we first compute the normal of the triangle
target_position = m.addAttribute("target_position", rows = 1, cols = 3)
target_position.updateValue([0, 16.0, 0.0])
tangent_energy_modifier = s0.addAttribute("tangent_energy_modifier", rows = 1, cols = 1)
tangent_energy_modifier.updateValue([TANGENT_ENERGY_MODIFIER]) # this is the modifier for the tangent energy
triangle_tangent_energy = tangent_energy_modifier * tangent_energy(pinball_triangle_positions.row(0), pinball_triangle_positions.row(1), pinball_triangle_positions.row(2), pinball_positions, target_position, pinball_velocities, DT) # this is the energy that tries to move the pinballs to the center of the triangle
pinballs.addAttribute("triangle_tangent_energy", computed_attribute = triangle_tangent_energy)

# we also add a wall energy at each edge of the triangle
# so that the pinballs do not go through the edges
center_distance_energy = triangle_center_distance_energy(pinball_triangle_positions.row(0), pinball_triangle_positions.row(1), pinball_triangle_positions.row(2), pinball_positions, pinball_velocities, DT, epsilon = 1.0)
center_distance_energy_modifier = s0.addAttribute("center_distance_energy_modifier", rows = 1, cols = 1)
center_distance_energy_modifier.updateValue([CENTER_DISTANCE_ENERGY_MODIFIER]) # this is the modifier for the wall energy
center_distance_energy = center_distance_energy_modifier * center_distance_energy
pinballs.addAttribute("center_distance_energy", computed_attribute = center_distance_energy)



minimizer0 = minimizer()
minimizer0.addEnergy(inertia)
minimizer0.addEnergy(pinball_collide_triangle)
minimizer0.addWrt([pinball_positions])
minimizer0.generateHessianAndGradient()

minimizer1 = minimizer()
minimizer1.addEnergy(edge_energy)
minimizer1.addEnergy(pinball_collide_triangle)
minimizer1.addEnergy(height_penalty)
minimizer1.addEnergy(triangle_tangent_energy)
# minimizer1.addEnergy(center_distance_energy)
minimizer1.addWrt([base_angle, first_segment_angle_sigmoid, second_segment_angle_sigmoid])

minimizer1.generateHessianAndGradient()
# exit()


import pyvista as pv
base_position_computed = base_position.compute().value.get().reshape(-1, 3)
first_segment_start_computed = base_position_computed.copy()
first_segment_start_computed[:, 1] += base_height.value.get()

first_segment_end_computed = first_segment_end_position.compute().value.get().reshape(-1, 3)
second_segment_end_computed = second_segment_end_position.compute().value.get().reshape(-1, 3)

points = np.stack((base_position_computed, first_segment_start_computed, first_segment_end_computed, second_segment_end_computed)).flatten().reshape(-1, 3)
lines = np.array([[2, 0, 3], [2, 1, 4], [2, 2, 5], [2, 3, 6], [2, 4, 7], [2, 5, 8], [2, 6, 9], [2, 7, 10], [2, 8, 11]])
# Create a PolyData object with lines
poly = pv.PolyData()
poly.points = points
poly.lines = lines

# Create a plotter
plotter = pv.Plotter(window_size=[1920, 1080])

# Add the lines to the plotter
plotter.add_mesh(poly, color="blue", line_width=3)
pinball_positions_computed = pinball_positions.compute().value.get().reshape(-1, 3)
pinball_radius_computed = radius.compute().value.get().reshape(-1, 1)

# Create PolyData for the triangle
triangle_poly = pv.PolyData()
triangle_poly.points = second_segment_end_computed
# Add a single triangle cell: note the indices are [0,1,2]
triangle_poly.faces = np.hstack([[3, 0, 1, 2]])
# Add the triangle mesh to the plotter
triangle_actor = plotter.add_mesh(triangle_poly, color="green", opacity=0.5)

# Add each sphere
sphere_actors = []
for center, radius in zip(pinball_positions_computed, pinball_radius_computed):
  sphere = pv.Sphere(radius=radius, center=(0,0,0))
  actor = plotter.add_mesh(sphere, color="blue", opacity=1)
  actor.SetPosition(center)
  sphere_actors.append(actor)


center = pv.Sphere(radius=radius, center=(0,16.0,0))
plotter.add_mesh(center, color="red", opacity=1)

# Show the plot window
# add a camera position
plotter.camera_position = [(85.9773971010733, 102.73793825237743, 49.58585230666707), (0, 16, 0), (-0.5625562346850076, 0.7528964923882753, -0.34158067065695963)]

all_line_segments = []
all_pinball_positions = []

plotter.show(interactive_update=True)
# exit()
STEP_SIZE = 0.1
for i in range(2000):
  all_line_segments.append(np.array(poly.points))
  all_pinball_positions.append(pinball_positions.value.get())
  pinball_last_positions.updateValue(pinball_positions.value.get())
  result0 = minimizer0.computeSolution(tolerance = 1e-16)
  result1 = minimizer1.computeSolution(tolerance = 1e-16)
  d_position = result0[0]
  d_base_angle, d_first_segment_angle_sigmoid, d_second_segment_angle_sigmoid = result1

  base_angle.updateValue(base_angle.value - STEP_SIZE * d_base_angle)
  first_segment_angle_sigmoid.updateValue(np.maximum(np.minimum(first_segment_angle_sigmoid.value.get() - STEP_SIZE * d_first_segment_angle_sigmoid.get(), 2.0), -2.0))
  second_segment_angle_sigmoid.updateValue(np.maximum(np.minimum(second_segment_angle_sigmoid.value.get() - STEP_SIZE * d_second_segment_angle_sigmoid.get(), 2.0), -2.0))

  pinball_positions.updateValue((pinball_positions.value - d_position))

  new_velocities = (pinball_positions.value.get() - pinball_last_positions.value.get()) / DT_VALUE
  if i % 300 == 0 and i != 0:
    new_velocities[0] += random.random() * 4 - 2.0
    new_velocities[1] += random.random() * 8
    new_velocities[2] += random.random() * 4 - 2.0
  pinball_velocities.updateValue(new_velocities)


  pinball_positions_cpu = pinball_positions.value.get().reshape(-1, 3)
  first_segment_end_computed = first_segment_end_position.compute().value.get().reshape(-1, 3)
  second_segment_end_computed = second_segment_end_position.compute().value.get().reshape(-1, 3)
  points = np.stack((base_position_computed, first_segment_start_computed, first_segment_end_computed, second_segment_end_computed)).flatten().reshape(-1, 3)

  poly.points = points
  triangle_poly.points = second_segment_end_computed

  for j in range(1):
    sphere_actors[j].SetPosition(pinball_positions_cpu[j, :])
  plotter.render()
  plotter.update()
  # plotter.screenshot(f"result_images/frame_{i:04d}.jpg")

# all_line_segments = np.array(all_line_segments)
# print(all_line_segments.shape)
# lines = np.array([[0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9], [7, 10], [8, 11]])
# triangle_face = np.array([[9, 10, 11]])
# pinball_positions = np.array(all_pinball_positions)
# print(pinball_positions.shape)
# # print(all_line_segments)

# np.savez("bouncing_balls.npz", arm_points=all_line_segments, lines=lines, triangle_face=triangle_face, pinball_positions=pinball_positions)
# print(all_line_segments[-1])
# print(all_pinball_positions[-1])
