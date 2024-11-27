from yasps import scene
from yasps import attribute
import numpy as np
import math
s0 = scene("scene0")
spring_system = s0.addMesh("spring_system")
spring_system.addAttribute("block_mass", rows = 1, cols = 1)

# we first add the top spring, which hangs from the ceiling
top_spring = spring_system.addPrimitive("top_spring", numInstances = 1)
top_spring.addAttribute("angle", rows = 1, cols = 1) # angle between each segment for the top most spring
top_spring.addAttribute("segment_length", rows = 1, cols = 1)
top_spring.addAttribute("segment_count", rows = 1, cols = 1)
top_spring.addAttribute("top_spring_end_point_y", computed_attribute = -0.5 - top_spring["angle"].sin() * top_spring["segment_length"] *  (top_spring["segment_count"] - 1.0) - 0.5) # the first -0.5 is distance from ceiling, the second -0.5 is from the end of the spring to the middle of the level

# we now add the lever
lever = spring_system.addPrimitive("lever", numInstances = 1)
lever_to_top_spring = lever.addConnectivity("lever_to_top_spring", top_spring, [0], 1) # define the relationship from the lever to the top spring
lever.addAttribute("middle_point_y", through = lever_to_top_spring, source = top_spring["top_spring_end_point_y"])
lever.addAttribute("length", rows = 1, cols = 1)
lever.addAttribute("angle", rows = 1, cols = 1)
lever.addAttribute("right_end_y", computed_attribute = lever["middle_point_y"] - 0.5 * lever["length"] * lever["angle"].sin())
lever.addAttribute("left_end_y", computed_attribute = lever["middle_point_y"] + 0.5 * lever["length"] * lever["angle"].sin())
lever.addAttribute("right_end_x", computed_attribute = 0.5 * lever["length"] * lever["angle"].cos())
lever.addAttribute("left_end_x", computed_attribute = -0.5 * lever["length"] * lever["angle"].cos())

# we now add the spring on the right side
right_spring = spring_system.addPrimitive("right_spring", numInstances = 1)
right_spring_to_lever = right_spring.addConnectivity("right_spring_to_lever", lever, [0], 1)
right_spring.addAttribute("angle", rows = 1, cols = 1) # angle between each segment for the right most spring
right_spring.addAttribute("segment_length", rows = 1, cols = 1)
right_spring.addAttribute("segment_count", rows = 1, cols = 1)
right_spring.addAttribute("end_x", through = right_spring_to_lever, source = lever["right_end_x"])
right_spring.addAttribute("top_y", through = right_spring_to_lever, source = lever["right_end_y"])
right_spring.addAttribute("end_y", computed_attribute = right_spring["top_y"] - right_spring["segment_length"] * (right_spring["segment_count"] - 1.0) * right_spring["angle"].sin())
right_spring.addAttribute("velocity", rows = 2, cols = 1) # for inertia
right_spring.addAttribute("last_position", rows = 2, cols = 1) # for inertia

# we now add the spring on the left side
left_spring = spring_system.addPrimitive("left_spring", numInstances = 1)
left_spring_to_lever = left_spring.addConnectivity("left_spring_to_lever", lever, [0], 1)
left_spring.addAttribute("angle", rows = 1, cols = 1) # angle between each segment for the left most spring
left_spring.addAttribute("segment_length", rows = 1, cols = 1)
left_spring.addAttribute("segment_count", rows = 1, cols = 1)
left_spring.addAttribute("end_x", through = left_spring_to_lever, source = lever["left_end_x"])
left_spring.addAttribute("top_y", through = left_spring_to_lever, source = lever["left_end_y"])
left_spring.addAttribute("end_y", computed_attribute = left_spring["top_y"] - left_spring["segment_length"] * (left_spring["segment_count"] - 1.0) * left_spring["angle"].sin())
left_spring.addAttribute("velocity", rows = 2, cols = 1) # for inertia
left_spring.addAttribute("last_position", rows = 2, cols = 1) # for inertia

def inertia(v0, vel, dt, x, mass):
  # v0 is the position we got before
  # vel is velocity
  # x is the position we are now
  x_target = v0 + vel * dt - attribute.to_array([attribute(float_value = 0.0), attribute(float_value = 9.8 * dt * dt)], rows = 2, cols = 1)
  return (0.5 * (x - x_target).transpose() * mass * (x - x_target))

BLOCK_MASS = 5.0
TOP_SPRING_REST_ANGLE = np.pi / 20
TOP_SPRING_INITIAL_ANGLE = np.pi / 4
TOP_SPRING_SEGMENT_COUNT = 9.0
TOP_SPRING_SEGMENT_LENGTH = 1.0
LEVER_INITIAL_ANGLE = 0.0
LEVER_LENGTH = 5.0
RIGHT_SPRING_REST_ANGLE = np.pi / 20
RIGHT_SPRING_INITIAL_ANGLE = np.pi / 4
RIGHT_SPRING_SEGMENT_COUNT = 7
RIGHT_SPRING_SEGMENT_LENGTH = 1.0
LEFT_SPRING_REST_ANGLE = np.pi / 20
LEFT_SPRING_INITIAL_ANGLE = np.pi / 4
LEFT_SPRING_SEGMENT_COUNT = 5
LEFT_SPRING_SEGMENT_LENGTH = 1.0
DT = 0.1

top_spring["angle"].updateValue([TOP_SPRING_INITIAL_ANGLE])
top_spring["segment_length"].updateValue([TOP_SPRING_SEGMENT_LENGTH])
top_spring["segment_count"].updateValue([TOP_SPRING_SEGMENT_COUNT])
lever["angle"].updateValue([LEVER_INITIAL_ANGLE])
lever["length"].updateValue([LEVER_LENGTH])
right_spring["angle"].updateValue([RIGHT_SPRING_INITIAL_ANGLE])
right_spring["segment_length"].updateValue([RIGHT_SPRING_SEGMENT_LENGTH])
right_spring["segment_count"].updateValue([RIGHT_SPRING_SEGMENT_COUNT])
# initialize condition for inertia
right_spring["velocity"].updateValue([0.0, 0.0])
right_spring["last_position"].updateValue([right_spring["end_x"].compute().value.get()[0], right_spring["end_y"].compute().value.get()[0]])


left_spring["angle"].updateValue([LEFT_SPRING_INITIAL_ANGLE])
left_spring["segment_length"].updateValue([LEFT_SPRING_SEGMENT_LENGTH])
left_spring["segment_count"].updateValue([LEFT_SPRING_SEGMENT_COUNT])
# initialize condition for inertia
left_spring["velocity"].updateValue([0.0, 0.0])
left_spring["last_position"].updateValue([left_spring["end_x"].compute().value.get()[0], left_spring["end_y"].compute().value.get()[0]])

# ok now we need to add energy to the system
# the first is on the spring, spring has a rest angle, which should be followed
top_spring.addAttribute("top_spring_angle_energy", computed_attribute = (top_spring["angle"] - TOP_SPRING_REST_ANGLE) * (top_spring["angle"] - TOP_SPRING_REST_ANGLE) / (top_spring["angle"] * (np.pi / 2 - top_spring["angle"])) * (TOP_SPRING_SEGMENT_COUNT - 1.0))
right_spring.addAttribute("right_spring_angle_energy", computed_attribute = (right_spring["angle"] - RIGHT_SPRING_REST_ANGLE) * (right_spring["angle"] - RIGHT_SPRING_REST_ANGLE) / (right_spring["angle"] * (np.pi / 2 - right_spring["angle"])) * (RIGHT_SPRING_SEGMENT_COUNT - 1.0))
left_spring.addAttribute("left_spring_angle_energy", computed_attribute = (left_spring["angle"] - LEFT_SPRING_REST_ANGLE) * (left_spring["angle"] - LEFT_SPRING_REST_ANGLE) / (left_spring["angle"] * (np.pi / 2 - left_spring["angle"])) * (LEFT_SPRING_SEGMENT_COUNT - 1.0))

# add inertia for the mass at the end of the spring_system
right_spring.addAttribute("right_spring_inertia", computed_attribute = inertia(right_spring["last_position"], right_spring["velocity"], DT, attribute.to_array([right_spring["end_x"], right_spring["end_y"]], rows = 2, cols = 1), BLOCK_MASS))
left_spring.addAttribute("left_spring_inertia", computed_attribute = inertia(left_spring["last_position"], left_spring["velocity"], DT, attribute.to_array([left_spring["end_x"], left_spring["end_y"]], rows = 2, cols = 1), BLOCK_MASS))

s0.addEnergy(top_spring["top_spring_angle_energy"])
s0.addEnergy(right_spring["right_spring_angle_energy"])
s0.addEnergy(left_spring["left_spring_angle_energy"])
s0.addEnergy(right_spring["right_spring_inertia"])
s0.addEnergy(left_spring["left_spring_inertia"])
s0.addMinimizeTarget([top_spring["angle"], right_spring["angle"], left_spring["angle"], lever["angle"]])
# s0.addMinimizeTarget([top_spring["angle"], right_spring["angle"]])




import matplotlib.pyplot as plt
plt.ion()
fig, ax = plt.subplots(figsize=(6, 8))
ceiling = ax.plot([-100, 100], [0, 0], linestyle='-', color='black')
ceiling_line,  = ax.plot([0, 0], [0, -0.5], linestyle='-', color='black')
top_spring_lines,  = ax.plot([], [], linestyle = '-', color = 'black')
lever_line, = ax.plot([], [], linestyle='-', color = 'black')
right_spring_lines, = ax.plot([], [], linestyle='-', color = 'black')
left_spring_lines, = ax.plot([], [], linestyle='-', color = 'black')
right_mass_box, = ax.plot([], [], linestyle='-', color = 'black')
left_mass_box, = ax.plot([], [], linestyle='-', color = 'black')
plt.axis('off')
plt.gca().set_aspect('equal', adjustable='box')
ax.set_xlim(-15, 15)
ax.set_ylim(-20, 10)
plt.show()

def update_plot():
  # update the top spring
  top_spring_angle = top_spring["angle"].compute().value.get()[0]
  cos_angle = math.cos(top_spring_angle)
  sin_angle = math.sin(top_spring_angle)
  L = TOP_SPRING_SEGMENT_LENGTH
  N = int(TOP_SPRING_SEGMENT_COUNT)
  delta_xs = np.zeros(N + 1)
  delta_ys = np.zeros(N + 1)
  delta_xs[0] = cos_angle * L / 2.0
  delta_ys[0] = -sin_angle * L / 2.0
  indices = np.arange(1, N - 1)
  directions = (-1) ** indices
  delta_xs[1:N-1] = cos_angle * L * directions
  delta_ys[1:N-1] = -sin_angle * L
  delta_xs[N - 1] = cos_angle * L / 2.0
  delta_ys[N - 1] = -sin_angle * L / 2.0
  delta_xs[N] = 0.0
  delta_ys[N] = -0.5
  x0 = 0.0  # Starting x-coordinate
  y0 = -0.5  # Starting y-coordinate
  top_spring_xs = x0 + np.concatenate(([0.0], np.cumsum(delta_xs)))
  top_spring_ys = y0 + np.concatenate(([0.0], np.cumsum(delta_ys)))
  top_spring_lines.set_data(top_spring_xs, top_spring_ys)

  # update the lever
  lever_right_x = lever["right_end_x"].compute().value.get()[0]
  lever_right_y = lever["right_end_y"].compute().value.get()[0]
  lever_left_x = lever["left_end_x"].compute().value.get()[0]
  lever_left_y = lever["left_end_y"].compute().value.get()[0]
  lever_line.set_data([lever_right_x, lever_left_x], [lever_right_y, lever_left_y])

  # update the right spring
  right_spring_angle = right_spring["angle"].compute().value.get()[0]
  cos_angle = math.cos(right_spring_angle)
  sin_angle = math.sin(right_spring_angle)
  L = RIGHT_SPRING_SEGMENT_LENGTH
  N = int(RIGHT_SPRING_SEGMENT_COUNT)
  delta_xs = np.zeros(N + 1)
  delta_ys = np.zeros(N + 1)
  delta_xs[0] = cos_angle * L / 2.0
  delta_ys[0] = -sin_angle * L / 2.0
  indices = np.arange(1, N - 1)
  directions = (-1) ** indices
  delta_xs[1:N-1] = cos_angle * L * directions
  delta_ys[1:N-1] = -sin_angle * L
  delta_xs[N - 1] = cos_angle * L / 2.0
  delta_ys[N - 1] = -sin_angle * L / 2.0
  delta_xs[N] = 0.0
  delta_ys[N] = -0.5
  x0 = lever_right_x  # Starting x-coordinate
  y0 = lever_right_y  # Starting y-coordinate
  right_spring_xs = x0 + np.concatenate(([0.0], np.cumsum(delta_xs)))
  right_spring_ys = y0 + np.concatenate(([0.0], np.cumsum(delta_ys)))
  right_spring_lines.set_data(right_spring_xs, right_spring_ys)
  right_mass_box.set_data([right_spring_xs[-1] - 0.5, right_spring_xs[-1] + 0.5, right_spring_xs[-1] + 0.5, right_spring_xs[-1] - 0.5, right_spring_xs[-1] - 0.5], [right_spring_ys[-1], right_spring_ys[-1], right_spring_ys[-1] - 0.6, right_spring_ys[-1] - 0.6, right_spring_ys[-1]])

  # update the left spring
  left_spring_angle = left_spring["angle"].compute().value.get()[0]
  cos_angle = math.cos(left_spring_angle)
  sin_angle = math.sin(left_spring_angle)
  L = LEFT_SPRING_SEGMENT_LENGTH
  N = int(LEFT_SPRING_SEGMENT_COUNT)
  delta_xs = np.zeros(N + 1)
  delta_ys = np.zeros(N + 1)
  delta_xs[0] = cos_angle * L / 2.0
  delta_ys[0] = -sin_angle * L / 2.0
  indices = np.arange(1, N - 1)
  directions = (-1) ** indices
  delta_xs[1:N-1] = cos_angle * L * directions
  delta_ys[1:N-1] = -sin_angle * L
  delta_xs[N - 1] = cos_angle * L / 2.0
  delta_ys[N - 1] = -sin_angle * L / 2.0
  delta_xs[N] = 0.0
  delta_ys[N] = -0.5
  x0 = lever_left_x  # Starting x-coordinate
  y0 = lever_left_y  # Starting y-coordinate
  left_spring_xs = x0 + np.concatenate(([0.0], np.cumsum(delta_xs)))
  left_spring_ys = y0 + np.concatenate(([0.0], np.cumsum(delta_ys)))
  left_spring_lines.set_data(left_spring_xs, left_spring_ys)
  left_mass_box.set_data([left_spring_xs[-1] - 0.5, left_spring_xs[-1] + 0.5, left_spring_xs[-1] + 0.5, left_spring_xs[-1] - 0.5, left_spring_xs[-1] - 0.5], [left_spring_ys[-1], left_spring_ys[-1], left_spring_ys[-1] - 0.6, left_spring_ys[-1] - 0.6, left_spring_ys[-1]])


  # update the right spring
  fig.canvas.draw()
  fig.canvas.flush_events()
iteration = 0

right_spring_mass_position_last = right_spring["last_position"].compute().value.get()
left_spring_mass_position_last = left_spring["last_position"].compute().value.get()
while(iteration < 1500):
  # print(right_spring["right_spring_inertia"].compute().value.get())
  # exit(0)
  result = s0.minimizeEnergy()
  # first we update the last position
  right_spring["last_position"].updateValue([right_spring["end_x"].compute().value.get()[0], right_spring["end_y"].compute().value.get()[0]])
  left_spring["last_position"].updateValue([left_spring["end_x"].compute().value.get()[0], left_spring["end_y"].compute().value.get()[0]])

  if (iteration % int(1 / DT) == 0):
    # update the velocity
    right_spring_new_position = right_spring["last_position"].compute().value.get()
    left_spring_new_position = left_spring["last_position"].compute().value.get()
    right_spring_vel = (right_spring_new_position - right_spring_mass_position_last) / DT
    left_spring_vel = (left_spring_new_position - left_spring_mass_position_last) / DT
    right_spring["velocity"].updateValue(right_spring_vel)
    left_spring["velocity"].updateValue(left_spring_vel)
    right_spring_mass_position_last = right_spring_new_position
    left_spring_mass_position_last = left_spring_new_position

  top_spring["angle"].updateValue(top_spring["angle"].compute().value.get()[0] - DT * result[0].get()[0])
  right_spring["angle"].updateValue(right_spring["angle"].compute().value.get()[0] - DT * result[1].get()[0])
  left_spring["angle"].updateValue(left_spring["angle"].compute().value.get()[0] - DT * result[2].get()[0])
  lever["angle"].updateValue(lever["angle"].compute().value.get()[0] - DT * result[3].get()[0])
  update_plot()
  iteration += 1
  plt.savefig(f'plots/frame_{iteration:04d}.png', dpi=600)
