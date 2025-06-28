from yasps import scene
from yasps import attribute
import math
import numpy as np

# define PI
PI = 3.141592653589793
s0 = scene("scene0")
m = s0.addMesh("mesh")

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
base_height.updateValue([5.0, 5.0, 5.0]) # the height of the base

first_segment_length = arms.addAttribute("first_segment_length", rows = 1, cols = 1) # the length of the first segment
first_segment_length.updateValue([4.0, 4.0, 4.0])

first_segment_angle = arms.addAttribute("first_segment_angle", rows = 1, cols = 1) # the angle of the first segment
first_segment_angle.updateValue([PI / 4.0, PI / 4.0, PI / 4.0]) # the angle of the first segment, we allow it to rotate around z axis

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
# print(first_segment_local_x_in_base.compute().value.get())
# print(first_segment_local_z_in_base.compute().value.get())

# # now we have the x y z, we rotate it
first_segment_global_x = first_segment_local_x_in_base * base_angle_from_center.cos() + first_segment_local_z_in_base * base_angle_from_center.sin()
first_segment_global_y = first_segment_local_y + base_height
first_segment_global_z = first_segment_local_x_in_base * base_angle_from_center.sin() - first_segment_local_z_in_base * base_angle_from_center.cos()
# print(first_segment_global_x.compute().value.get())
# print(first_segment_global_z.compute().value.get())


# add the attribute
first_segment_end_position = arms.addAttribute("first_segment_end_position", computed_attribute=attribute.to_array([first_segment_global_x, first_segment_global_y, first_segment_global_z], rows=3, cols=1))


import pyvista as pv
base_position_computed = base_position.compute().value.get().reshape(-1, 3)
first_segment_start_computed = base_position_computed.copy()
first_segment_start_computed[:, 1] += base_height.value.get()

first_segment_end_computed = first_segment_end_position.compute().value.get().reshape(-1, 3)

# print(base_position_computed)
# print(first_segment_start_computed)
# print(first_segment_end_computed)

points = np.stack((base_position_computed, first_segment_start_computed, first_segment_end_computed)).flatten().reshape(-1, 3)
lines = np.array([[2, 0, 3], [2, 1, 4], [2, 2, 5], [2, 3, 6], [2, 4, 7], [2, 5, 8]])
# Create a PolyData object with lines
poly = pv.PolyData()
poly.points = points
poly.lines = lines

# Create a plotter
plotter = pv.Plotter()

# Add the lines to the plotter
plotter.add_mesh(poly, color="blue", line_width=3)

# Show the plot window
plotter.show()
