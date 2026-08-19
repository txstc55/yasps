import os
import shutil
import subprocess
import time

import numpy as np
import pycuda.gpuarray as gpuarray
import pyvista as pv

from yasps import scene
from helpers import extract_surface_triangles, inertia, moving_energy, stable_neo_hookean


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REST_MESH_PATH = os.path.join(SCRIPT_DIR, "../data/solid_double_clamp_block_tet.msh")
PINCHED_MESH_PATH = os.path.join(SCRIPT_DIR, "../data/solid_double_clamp_block_tet_pinched.msh")
OUTPUT_DIRECTORY = os.path.join(SCRIPT_DIR, "outputs")
FRAME_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, "frames")
VIDEO_PATH = os.path.join(OUTPUT_DIRECTORY, "double_clamp_material_optimization.mp4")

DT_VALUE = 1.0 / 60.0
NUM_FRAMES = 60
VIDEO_FPS = 60
DENSITY_VALUE = 1000.0
YOUNG_VALUE = 150000.0
POISSON_VALUE = 0.45
PIN_STIFFNESS_VALUE = 25000.0
CONTROL_X_MAX = -0.03
SOLVER_TOLERANCE = 1e-5
MOTION_TOLERANCE = 1e-6
MAX_NEWTON_ITERATIONS = 30
MAX_CG_ITERATIONS = 20000
MAX_LINE_SEARCH_STEPS = 12


##################################################################
## Load the rest and pinched tetrahedral meshes
##################################################################
rest_grid = pv.read(REST_MESH_PATH)
pinched_grid = pv.read(PINCHED_MESH_PATH)
if pv.CellType.TETRA not in rest_grid.cells_dict or pv.CellType.TETRA not in pinched_grid.cells_dict:
  raise ValueError("Both input meshes must contain linear tetrahedra.")

rest_position_values = np.asarray(rest_grid.points, dtype = np.float64) * 1e-3
pinched_position_values = np.asarray(pinched_grid.points, dtype = np.float64) * 1e-3
tetrahedron_indices = np.asarray(rest_grid.cells_dict[pv.CellType.TETRA], dtype = np.uint32)
pinched_tetrahedron_indices = np.asarray(pinched_grid.cells_dict[pv.CellType.TETRA], dtype = np.uint32)
if rest_position_values.shape != pinched_position_values.shape:
  raise ValueError("The rest and pinched meshes have different vertex counts.")
if not np.array_equal(tetrahedron_indices, pinched_tetrahedron_indices):
  raise ValueError("The rest and pinched meshes do not have matching tetrahedra.")

controlled_indices = np.flatnonzero(rest_position_values[:, 0] < CONTROL_X_MAX).astype(np.uint32)
if controlled_indices.size == 0:
  raise ValueError("No vertices satisfy x < -0.03 m.")
controlled_rest_positions = rest_position_values[controlled_indices]
controlled_target_positions = pinched_position_values[controlled_indices]
target_displacements = controlled_target_positions - controlled_rest_positions


##################################################################
## Compute volume-lumped vertex masses
##################################################################
tetrahedron_positions = rest_position_values[tetrahedron_indices]
edge_matrices = np.stack([tetrahedron_positions[:, 1] - tetrahedron_positions[:, 0], tetrahedron_positions[:, 2] - tetrahedron_positions[:, 0], tetrahedron_positions[:, 3] - tetrahedron_positions[:, 0]], axis = 2)
tetrahedron_volumes = np.abs(np.linalg.det(edge_matrices)) / 6.0
if np.any(tetrahedron_volumes <= 0.0):
  raise ValueError("The rest mesh contains a degenerate tetrahedron.")
vertex_masses = np.zeros(rest_position_values.shape[0], dtype = np.float64)
np.add.at(vertex_masses, tetrahedron_indices.reshape(-1), np.repeat(0.25 * DENSITY_VALUE * tetrahedron_volumes, 4))
total_volume = float(tetrahedron_volumes.sum())
total_mass = float(vertex_masses.sum())

print(f"Loaded {rest_position_values.shape[0]} vertices and {tetrahedron_indices.shape[0]} tetrahedra.")
print(f"Driving {controlled_indices.size} vertices with x < {CONTROL_X_MAX:.2f} m; maximum target displacement={np.linalg.norm(target_displacements, axis = 1).max():.6f} m.")
print(f"Material: E={YOUNG_VALUE:.1f} Pa, nu={POISSON_VALUE:.3f}, density={DENSITY_VALUE:.1f} kg/m^3, pin stiffness={PIN_STIFFNESS_VALUE:.1f} N/m.")
print(f"Rest volume={total_volume:.8e} m^3, total mass={total_mass:.8e} kg, gravity=0.")


##################################################################
## Construct the YASPS scene
##################################################################
simulation = scene("double_clamp_material_optimization")
dt = simulation.addConstant("dt", rows = 1, cols = 1)
dt.updateValue([DT_VALUE])

double_clamp = simulation.addMesh("double_clamp")
young_modulus = double_clamp.addConstant("young_modulus", rows = 1, cols = 1)
young_modulus.updateValue([YOUNG_VALUE])
poisson_ratio = double_clamp.addConstant("poisson_ratio", rows = 1, cols = 1)
poisson_ratio.updateValue([POISSON_VALUE])
mu_lame = young_modulus / (2.0 * (1.0 + poisson_ratio))
lambda_lame = young_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))

vertices = double_clamp.addPrimitive("vertices", numInstances = rest_position_values.shape[0])
position = vertices.addAttribute("position", rows = 3, cols = 1)
rest_position = vertices.addConstant("rest_position", rows = 3, cols = 1)
last_position = vertices.addConstant("last_position", rows = 3, cols = 1)
last_last_position = vertices.addConstant("last_last_position", rows = 3, cols = 1)
mass = vertices.addConstant("mass", rows = 1, cols = 1)
velocity = (last_position - last_last_position) / DT_VALUE

position.updateValue(rest_position_values.reshape(-1))
rest_position.updateValue(rest_position_values.reshape(-1))
last_position.updateValue(rest_position_values.reshape(-1))
last_last_position.updateValue(rest_position_values.reshape(-1))
mass.updateValue(vertex_masses)

tetrahedra = double_clamp.addPrimitive("tetrahedra", numInstances = tetrahedron_indices.shape[0])
tetrahedra_to_vertices = tetrahedra.addConnectivity("tetrahedra_to_vertices", vertices, tetrahedron_indices, 4)
tetrahedron_position = tetrahedra.addAttribute("positions", through = tetrahedra_to_vertices, source = position)
tetrahedron_rest_position = tetrahedra.addAttribute("rest_positions", through = tetrahedra_to_vertices, source = rest_position)

controlled_vertices = double_clamp.addPrimitive("controlled_vertices", numInstances = controlled_indices.size)
controlled_to_vertices = controlled_vertices.addConnectivity("controlled_to_vertices", vertices, controlled_indices.reshape((-1, 1)), 1)
controlled_position = controlled_vertices.addAttribute("position", through = controlled_to_vertices, source = position)
target_position = controlled_vertices.addConstant("target_position", rows = 1, cols = 3)
target_position.updateValue(controlled_target_positions.reshape(-1))
target_stiffness = controlled_vertices.addConstant("target_stiffness_per_vertex", rows = 1, cols = 1)
target_stiffness.updateValue(np.full(controlled_indices.size, PIN_STIFFNESS_VALUE, dtype = np.float64))


##################################################################
## Add the elastic, zero-gravity inertia, and target energies
##################################################################
elastic_energy = stable_neo_hookean(tetrahedron_rest_position, tetrahedron_position, mu_lame, lambda_lame, dt)
tetrahedra.addAttribute("stable_neo_hookean_energy", computed_attribute = elastic_energy)
inertia_energy = inertia(last_position, velocity, DT_VALUE, position, mass)
vertices.addAttribute("zero_gravity_inertia_energy", computed_attribute = inertia_energy)
target_energy = moving_energy(controlled_position, target_position, dt, target_stiffness)
controlled_vertices.addAttribute("target_position_penalty", computed_attribute = target_energy)

simulation.addEnergy(elastic_energy, projection_method = 1)
simulation.addEnergy(inertia_energy, projection_method = -1)
simulation.addEnergy(target_energy, projection_method = -1)
simulation.addMinimizeTarget([position])


##################################################################
## Construct the PyVista surface and output paths
##################################################################
surface_triangles = extract_surface_triangles(tetrahedron_indices)
surface_indices = np.unique(surface_triangles).astype(np.uint32)
surface_local_indices = np.full(rest_position_values.shape[0], -1, dtype = np.int64)
surface_local_indices[surface_indices] = np.arange(surface_indices.size)
surface_triangles_local = surface_local_indices[surface_triangles]
if np.any(surface_triangles_local < 0):
  raise RuntimeError("A surface triangle contains a non-surface vertex.")
surface_cells = np.hstack([np.full((surface_triangles.shape[0], 1), 3, dtype = np.uint32), surface_triangles_local.astype(np.uint32)])
surface_poly = pv.PolyData(rest_position_values[surface_indices], surface_cells)
controlled_surface_indices = np.intersect1d(surface_indices, controlled_indices, assume_unique = True)
target_points = pv.PolyData(pinched_position_values[controlled_surface_indices])

os.makedirs(FRAME_DIRECTORY, exist_ok = True)
for old_frame in os.listdir(FRAME_DIRECTORY):
  if old_frame.startswith("frame_") and old_frame.endswith(".png"):
    os.unlink(os.path.join(FRAME_DIRECTORY, old_frame))

plotter = pv.Plotter(window_size = [1920, 1080], off_screen = True)
plotter.set_background("#f4f5f2")
plotter.add_mesh(surface_poly, color = "#6aa879", smooth_shading = True, show_edges = False, specular = 0.2)
plotter.add_points(target_points, color = "#c94040", point_size = 5, render_points_as_spheres = True)
plotter.add_text("Double clamp", name = "frame_status", position = "upper_left", font_size = 16, color = "#202124")
plotter.camera_position = [(0.0, 0.0, 0.30), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
plotter.enable_parallel_projection()
plotter.camera.parallel_scale = 0.034
plotter.show(auto_close = False)


##################################################################
## Run the 60-frame forward simulation
##################################################################
simulation_start = time.time()
try:
  for frame in range(NUM_FRAMES):
    ramp_phase = (frame + 1) / NUM_FRAMES
    pin_progress = 0.5 - 0.5 * np.cos(np.pi * ramp_phase)
    active_targets = controlled_rest_positions + pin_progress * (controlled_target_positions - controlled_rest_positions)
    target_position.updateValue(active_targets.reshape(-1))

    last_last_position.updateValue(last_position.value, deepCopy = True)
    last_position.updateValue(position.value, deepCopy = True)

    converged = False
    frame_start = time.time()
    for newton_iteration in range(MAX_NEWTON_ITERATIONS):
      energy_before = simulation.computeTotalEnergy()
      position_before = position.value.copy()
      displacement = simulation.minimizeEnergy(tolerance = SOLVER_TOLERANCE, maxIterations = MAX_CG_ITERATIONS)[0]
      max_displacement = float(gpuarray.max(abs(displacement)).get())

      step_taken = 1.0
      accepted = False
      energy_after = energy_before
      for _ in range(MAX_LINE_SEARCH_STEPS):
        position.updateValue(position_before - step_taken * displacement, deepCopy = True)
        energy_after = simulation.computeTotalEnergy()
        if np.isfinite(energy_after) and energy_after <= energy_before + 1e-12 * max(1.0, abs(energy_before)):
          accepted = True
          break
        step_taken *= 0.5

      if not accepted:
        position.updateValue(position_before, deepCopy = True)
        if max_displacement > 10.0 * MOTION_TOLERANCE:
          raise RuntimeError("Newton line search failed to find an energy-decreasing step.")
        step_taken = 0.0
        energy_after = energy_before

      print(f"frame={frame:02d} newton={newton_iteration:02d} step={step_taken:.6f} max_dx={max_displacement:.3e} energy={energy_before:.8e}->{energy_after:.8e}")
      if step_taken * max_displacement < MOTION_TOLERANCE:
        converged = True
        break

    if not converged:
      print(f"Warning: frame {frame} reached the {MAX_NEWTON_ITERATIONS}-iteration Newton cap.")

    current_position_values = position.value.get().reshape((-1, 3))
    controlled_error = current_position_values[controlled_indices] - active_targets
    maximum_pin_error = float(np.linalg.norm(controlled_error, axis = 1).max())
    surface_poly.points = current_position_values[surface_indices]
    surface_poly.compute_normals(inplace = True)
    plotter.add_text(f"Double clamp  |  t = {(frame + 1) * DT_VALUE:.3f} s", name = "frame_status", position = "upper_left", font_size = 16, color = "#202124")
    plotter.render()
    plotter.screenshot(os.path.join(FRAME_DIRECTORY, f"frame_{frame:04d}.png"))
    print(f"completed frame={frame:02d} seconds={time.time() - frame_start:.3f} pin_progress={pin_progress:.6f} max_pin_error={maximum_pin_error:.3e} m")
finally:
  plotter.close()

print(f"Finished {NUM_FRAMES} frames in {time.time() - simulation_start:.2f} s.")


##################################################################
## Encode the saved frames
##################################################################
ffmpeg = shutil.which("ffmpeg")
if ffmpeg is None:
  raise RuntimeError("ffmpeg is required to encode the saved PNG frames.")
expected_frames = [os.path.join(FRAME_DIRECTORY, f"frame_{frame:04d}.png") for frame in range(NUM_FRAMES)]
missing_frames = [frame_path for frame_path in expected_frames if not os.path.exists(frame_path)]
if missing_frames:
  raise RuntimeError(f"Cannot encode video: {len(missing_frames)} frame(s) are missing.")
ffmpeg_command = [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(VIDEO_FPS), "-start_number", "0", "-i", os.path.join(FRAME_DIRECTORY, "frame_%04d.png"), "-frames:v", str(NUM_FRAMES), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", VIDEO_PATH]
subprocess.run(ffmpeg_command, check = True)
print(f"Saved frames: {FRAME_DIRECTORY}")
print(f"Saved video: {VIDEO_PATH}")
