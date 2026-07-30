"""
Implicit particle simulation and inverse-simulation Jacobian benchmark.

The forward step minimizes an implicit Euler inertia/spring/gravity energy.
At every step, the benchmark materializes

    d² E / d(position) d(previous_position)

and applies both the matrix and its transpose, matching the two products used
by forward sensitivities and adjoints.
"""

import argparse
import statistics
import time

import numpy as np
import pycuda.driver as cuda

from yasps import attribute, differentiator, scene, vector


def make_simulation(num_particles):
  model = scene("inverse_particle_simulation")
  mesh = model.addMesh("mesh")
  particles = mesh.addPrimitive("particles", numInstances=num_particles)

  position = particles.addAttribute("position", rows=3, cols=1)
  previous_position = particles.addConstant(
    "previous_position", rows=3, cols=1
  )
  velocity = particles.addConstant("velocity", rows=3, cols=1)
  rest_position = particles.addConstant("rest_position", rows=3, cols=1)
  mass = particles.addConstant("mass", rows=1, cols=1)

  side = int(np.ceil(np.sqrt(num_particles)))
  initial = np.zeros((num_particles, 3), dtype=np.float64)
  initial[:, 0] = np.arange(num_particles) % side
  initial[:, 2] = np.arange(num_particles) // side
  initial *= 0.025
  initial[:, 1] = 1.0
  position.updateValue(initial)
  previous_position.updateValue(initial)
  rest_position.updateValue(initial)
  velocity.updateValue(np.zeros_like(initial))
  mass.updateValue(np.linspace(0.8, 1.2, num_particles))

  dt = 1.0 / 120.0
  stiffness = 35.0
  gravity = attribute.to_array([0.0, -9.81, 0.0], rows=3, cols=1)
  predicted = previous_position + dt * velocity
  displacement = position - predicted
  inertia = 0.5 * mass / (dt * dt) * displacement.dot(displacement)
  spring_delta = position - rest_position
  spring = 0.5 * stiffness * spring_delta.dot(spring_delta)
  gravity_potential = -mass * gravity.dot(position)
  energy = particles.addAttribute(
    "implicit_energy",
    computed_attribute=inertia + spring + gravity_potential
  )

  model.addEnergy(energy, projection_method=-1)
  model.addMinimizeTarget([position])
  mixed = differentiator().diff2(
    [energy],
    [position],
    [previous_position],
    compress_coordinates=False
  )
  return (
    model,
    position,
    previous_position,
    velocity,
    mass,
    mixed,
    dt
  )


def timed_gpu(callable_value):
  cuda.Context.synchronize()
  start = time.perf_counter()
  result = callable_value()
  cuda.Context.synchronize()
  return result, (time.perf_counter() - start) * 1000.0


def run(num_particles, steps, warmup):
  (
    model,
    position,
    previous_position,
    velocity,
    mass,
    mixed,
    dt
  ) = make_simulation(num_particles)
  right = vector(3 * num_particles)
  left = vector(3 * num_particles)
  rng = np.random.default_rng(7)
  right.updateValue(rng.standard_normal(right.size))
  left.updateValue(rng.standard_normal(left.size))

  mixed.compute()
  mixed.matVecProduct(right)
  mixed.transposeMatVecProduct(left)

  solve_times = []
  jacobian_times = []
  spmv_times = []
  transpose_spmv_times = []
  for step in range(steps + warmup):
    old_position = position.value.copy()
    previous_position.updateValue(old_position, deepCopy=True)
    direction, solve_ms = timed_gpu(
      lambda: model.minimizeEnergy(tolerance=1.0e-10, maxIterations=200)
    )
    position.updateValue(position.value - direction[0], deepCopy=True)
    velocity.updateValue(
      (position.value - old_position) / dt,
      deepCopy=True
    )

    _, jacobian_ms = timed_gpu(mixed.compute)
    _, spmv_ms = timed_gpu(lambda: mixed.matVecProduct(right))
    _, transpose_ms = timed_gpu(
      lambda: mixed.transposeMatVecProduct(left)
    )
    if step >= warmup:
      solve_times.append(solve_ms)
      jacobian_times.append(jacobian_ms)
      spmv_times.append(spmv_ms)
      transpose_spmv_times.append(transpose_ms)

  expected_diagonal = -np.repeat(
    mass.value.get() / (dt * dt), 3
  )
  dense_diagonal = np.diag(mixed.toDense())
  np.testing.assert_allclose(dense_diagonal, expected_diagonal)

  print(f"particles: {num_particles}")
  print(f"steps: {steps}")
  print(
    "implicit solve: "
    f"{statistics.median(solve_times):.4f} ms median, "
    f"{statistics.mean(solve_times):.4f} ms mean"
  )
  print(
    "second-order Jacobian compute: "
    f"{statistics.median(jacobian_times):.4f} ms median, "
    f"{statistics.mean(jacobian_times):.4f} ms mean"
  )
  print(
    "Jacobian SpMV: "
    f"{statistics.median(spmv_times):.4f} ms median, "
    f"{statistics.mean(spmv_times):.4f} ms mean"
  )
  print(
    "transpose Jacobian SpMV: "
    f"{statistics.median(transpose_spmv_times):.4f} ms median, "
    f"{statistics.mean(transpose_spmv_times):.4f} ms mean"
  )


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--particles", type=int, default=256)
  parser.add_argument("--steps", type=int, default=20)
  parser.add_argument("--warmup", type=int, default=3)
  arguments = parser.parse_args()
  run(arguments.particles, arguments.steps, arguments.warmup)
