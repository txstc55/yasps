from yasps import scene
from yasps import attribute
import numpy as np

NUM_PINS_PER_DIRECTION = 200
SQUARE_SIZE = 20.0
DT = 0.1
# get the pin's initial position
locations = []
for i in range(NUM_PINS_PER_DIRECTION):
  for j in range(NUM_PINS_PER_DIRECTION):
    locations.append([i * SQUARE_SIZE / NUM_PINS_PER_DIRECTION - SQUARE_SIZE / 2, 0.0, j * SQUARE_SIZE / NUM_PINS_PER_DIRECTION - SQUARE_SIZE / 2])


s0 = scene("scene0")
pb = s0.addMesh("pin_and_ball")
pb.addAttribute("ball_center", rows = 3, cols = 1)
pb.addAttribute("ball_radius", rows = 1, cols = 1)
pb["ball_center"].updateValue([0.0, 3.0, 0.0])
pb["ball_radius"].updateValue([1.0])
pb.addAttribute("ball_center_before", rows = 3, cols = 1)
pb["ball_center_before"].updateValue(pb["ball_center"].value, deepCopy = True)
pb.addAttribute("ball_velocity", rows = 3, cols = 1)
pb["ball_velocity"].updateValue([0.0, 0.0, 0.0])
pb.addAttribute("ball_mass", rows = 1, cols = 1)
pb["ball_mass"].updateValue([0.1])
pb.addAttribute("ball_target", rows = 3, cols = 1)
pb["ball_target"].updateValue([5., 2.0, 5.0])


# now we add pins
pins = pb.addPrimitive("pins", numInstances = NUM_PINS_PER_DIRECTION * NUM_PINS_PER_DIRECTION)
pins.addAttribute("base_position", rows = 3, cols = 1)
pins["base_position"].updateValue(np.array(locations))

# pin can only move in the y direction
pyd = pins.addAttribute("y_displacement", rows = 1, cols = 1)
pins["y_displacement"].updateValue([1.0] * NUM_PINS_PER_DIRECTION * NUM_PINS_PER_DIRECTION)
# now we add the tangent point energy

def repulsive_energy(pin_base, pin_y_disp, center, radius, target):
  p0 = pin_base + attribute.to_array([0.0, pin_y_disp, 0.0], rows = 3, cols = 1) # this is the top of the pin
  direction = (center - p0) / (center - p0).norm()
  target_direction = (target - center) / (target - center).norm()
  return 1000.0 * (1.0 - (direction.dot(target_direction))) / (center - p0).norm().pow(0.001)

def repel_ball(pin_base, pin_y_disp, center, radius, dHat, kappa):
  p0 = pin_base + attribute.to_array([0.0, pin_y_disp, 0.0], rows = 3, cols = 1) # this is the top of the pin
  p1 = center + ((p0 - center) / (p0 - center).norm()) * radius
  distance = (p0 - p1).dot(p0 - p1).sqrt()
  distance = attribute.select(distance < attribute(float_value = dHat), distance, attribute(float_value = dHat))
  b = -((distance - dHat) * (distance - dHat)) * (distance / dHat).log()
  return kappa * b

pins.addAttribute("repulsive", computed_attribute = repulsive_energy(pins["base_position"], pins["y_displacement"], pb["ball_center"], pb["ball_radius"], pb["ball_target"]))
pins.addAttribute("position_penalty", computed_attribute = 10.0 * (pyd - 1.0) * (pyd - 1.0) / (pyd * (2.0 - pyd)))
pins.addAttribute("repel_ball_small", computed_attribute = repel_ball(pins["base_position"], pins["y_displacement"], pb["ball_center"], pb["ball_radius"], 0.1, 1.0))
pins.addAttribute("repel_ball_large", computed_attribute = repel_ball(pins["base_position"], pins["y_displacement"], pb["ball_center"], pb["ball_radius"], 0.1, 2.0))

# ok now we add the inertia for the ball
def inertia(v0, vel, dt, x, mass):
  x_target = v0 + vel * dt - attribute.to_array([attribute(float_value = 0.0), attribute(float_value = 9.8 * dt * dt), attribute(float_value = 0.0)], rows = 3, cols = 1)
  return (0.5 * (x - x_target).transpose() * mass * (x - x_target))
pb.addAttribute('inertia', computed_attribute = inertia(pb['ball_center_before'], pb['ball_velocity'], DT, pb['ball_center'], pb['ball_mass']))
# pb.addAttribute('target_penalty', computed_attribute = 1000.0 * (pb['ball_center'] - pb['ball_target']) * (pb['ball_center'] - pb['ball_target']))

from yasps import minimizer
minimizer0 = minimizer()
minimizer1 = minimizer()

# ok for minimizer0, we want to optimize y's displacement so that it rolls the ball to the target
minimizer0.addEnergy(pins["repulsive"])
minimizer0.addEnergy(pins["position_penalty"])
minimizer0.addEnergy(pins["repel_ball_small"])
minimizer0.addWrt([pyd])

# for minimizer1, we will add inertia and the repel energy to move the ball only
minimizer1.addEnergy(pb['inertia'])
minimizer1.addEnergy(pins['repel_ball_large'])
# minimizer1.addEnergy(pins["repulsive"])
minimizer1.addWrt([pb['ball_center']])

minimizer0.generateHessianAndGradient()
minimizer1.generateHessianAndGradient()

# visualize the scene
import pyvista as pv
plotter = pv.Plotter()
ball = plotter.add_mesh(pv.Sphere(radius = 1.0, center = pb['ball_center'].value.get()), color = 'red', opacity = 0.3)
target = plotter.add_mesh(pv.Sphere(radius = 0.5, center = pb['ball_target'].value.get()), color = 'green', opacity = 1.0)

pins_mesh = []
# add pins
for i in range(NUM_PINS_PER_DIRECTION * NUM_PINS_PER_DIRECTION):
  pins_mesh.append(plotter.add_mesh(pv.Sphere(radius = 0.1, center = np.array(locations[i]), theta_resolution = 2, phi_resolution = 2), color = 'blue', opacity = 1.0))

camera_position = [
    (0, 30, 30),    # Camera position
    (0, 0, 0),    # Focal point (where the camera looks at)
    (0, 1, 0)     # View up direction
]
plotter.camera_position = camera_position
plotter.show(interactive_update=True)
ball_center_last = pb['ball_center'].value.copy()
iteration = 0
while iteration < 4000:
  # At the start of the iteration, set ball_center_before to the old position
  pb['ball_center_before'].updateValue(ball_center_last, deepCopy=True)
  # Now compute the solutions
  d_y = minimizer0.computeSolution()[0]
  d_center = minimizer1.computeSolution()[0]
  # Update pin and ball positions
  pins['y_displacement'].updateValue(pins['y_displacement'].value - DT * d_y)
  pb['ball_center'].updateValue(pb['ball_center'].value - DT * d_center)

  # Update velocity and record current position for next iteration's "before"
  pb['ball_velocity'].updateValue((pb['ball_center'].value - ball_center_last) / DT)
  ball_center_last = pb['ball_center'].value.copy()

  # Update visualization if needed
  if iteration % 1 == 0:
    y_displacements = pins['y_displacement'].value.get()
    for j in range(NUM_PINS_PER_DIRECTION * NUM_PINS_PER_DIRECTION):
      pins_mesh[j].SetPosition(np.array([0.0, y_displacements[j], 0.0]))
    ball.SetPosition(pb['ball_center'].value.get() - np.array([0.0, 3.0, 0.0]))
    plotter.update()

  iteration += 1
