import argparse
import json
import os
import shutil
import subprocess
import time


def query_system_gpu_memory():
  """Read whole-device memory before CUDA/YASPS creates this process's context."""
  nvidia_smi = shutil.which("nvidia-smi")
  if nvidia_smi is None:
    return None, None
  try:
    result = subprocess.run([nvidia_smi, "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    first_gpu = result.stdout.strip().splitlines()[0]
    used_mib, total_mib = (float(value.strip()) for value in first_gpu.split(","))
    return int(used_mib * 1024 ** 2), int(total_mib * 1024 ** 2)
  except (IndexError, OSError, subprocess.CalledProcessError, ValueError):
    return None, None


# Capture this before importing PyCUDA or YASPS so display/system allocations
# can be removed from the application's forward and backward memory numbers.
SYSTEM_GPU_BASELINE_USED_BYTES, SYSTEM_GPU_TOTAL_BYTES = query_system_gpu_memory()

import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pyvista as pv

from yasps import differentiator, scene, solver, vector
from helpers import extract_surface_triangles, inertia, moving_energy, stable_neo_hookean


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REST_MESH_PATH = os.path.join(SCRIPT_DIR, "../data/solid_double_clamp_block_tet.msh")
PINCHED_MESH_PATH = os.path.join(SCRIPT_DIR, "../data/solid_double_clamp_block_tet_pinched.msh")

NUM_FRAMES = 60
NUM_ADJOINT_STEPS = 100
DT_VALUE = 1.0 / 60.0
VIDEO_FPS = 60
DENSITY_VALUE = 1000.0
YOUNG_VALUE = 150000.0
POISSON_VALUE = 0.45
YOUNG_MIN_VALUE = 1000.0
YOUNG_MAX_VALUE = 1000000.0
POISSON_MIN_VALUE = 0.2
POISSON_MAX_VALUE = 0.48
YOUNG_LOGIT_DESIGN_LIMIT = 12.0
POISSON_LOGIT_DESIGN_LIMIT = 12.0
PIN_STIFFNESS_VALUE = 2500.0
TARGET_LOSS_WEIGHT_VALUE = 250000.0
CONTROL_X_MAX = -0.03
TARGET_X_MIN = 0.03
SOLVER_TOLERANCE = 1e-5
ADJOINT_SOLVER_TOLERANCE = 1e-8
MOTION_TOLERANCE = 1e-6
MAX_NEWTON_ITERATIONS = 300
MAX_CG_ITERATIONS = 20000
MAX_LINE_SEARCH_STEPS = 12
ADAM_LEARNING_RATE = 0.9
YOUNG_STEP_SIZE = 0.9
POISSON_STEP_SIZE = 0.9
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8


parser = argparse.ArgumentParser(description="Optimize one Young modulus and Poisson ratio per tetrahedron so pinching the left clamp drives the right clamp to its pinched target.")
parser.add_argument("--num-frames", type=int, default=NUM_FRAMES)
parser.add_argument("--adjoint-steps", type=int, default=NUM_ADJOINT_STEPS)
parser.add_argument("--adam-learning-rate", type=float, default=ADAM_LEARNING_RATE)
parser.add_argument("--young-step-size", type=float, default=YOUNG_STEP_SIZE)
parser.add_argument("--poisson-step-size", type=float, default=POISSON_STEP_SIZE)
parser.add_argument("--save-frames", action="store_true", help="Save each rendered frame and encode one MP4 per forward trajectory.")
parser.add_argument("--video-fps", type=int, default=VIDEO_FPS)
parser.add_argument("--video-every", type=int, default=10, help="Save a video every N inverse rounds; the baseline and final round are always saved.")
parser.add_argument("--checkpoint-every", type=int, default=10, help="Archive a restartable optimizer checkpoint every N inverse rounds; the final round is always archived.")
parser.add_argument("--no-gui", action="store_true")
parser.add_argument("--preview-only", action="store_true")
parser.add_argument("--gradient-check", action="store_true", help="Compare final-state adjoint derivatives with central finite differences and exit.")
parser.add_argument("--resume-checkpoint", type=str, default=None)
parser.add_argument("--output-directory", type=str, default=None)
args = parser.parse_args()

if args.num_frames <= 0:
  raise ValueError("--num-frames must be positive.")
if args.adjoint_steps < 0:
  raise ValueError("--adjoint-steps must be non-negative.")
if args.adam_learning_rate <= 0.0:
  raise ValueError("--adam-learning-rate must be positive.")
if args.young_step_size <= 0.0 or args.poisson_step_size <= 0.0:
  raise ValueError("The material step sizes must be positive.")
if args.video_fps <= 0:
  raise ValueError("--video-fps must be positive.")
if args.video_every <= 0:
  raise ValueError("--video-every must be positive.")
if args.checkpoint_every <= 0:
  raise ValueError("--checkpoint-every must be positive.")


##################################################################
## Small utilities for material design, diagnostics, and output
##################################################################
POISSON_BASE_FRACTION = (POISSON_VALUE - POISSON_MIN_VALUE) / (POISSON_MAX_VALUE - POISSON_MIN_VALUE)
POISSON_BASE_LOGIT = np.log(POISSON_BASE_FRACTION / (1.0 - POISSON_BASE_FRACTION))
YOUNG_BASE_FRACTION = (YOUNG_VALUE - YOUNG_MIN_VALUE) / (YOUNG_MAX_VALUE - YOUNG_MIN_VALUE)
YOUNG_BASE_LOGIT = np.log(YOUNG_BASE_FRACTION / (1.0 - YOUNG_BASE_FRACTION))


def poisson_values_from_design(design_parameters):
  shifted_logit = POISSON_BASE_LOGIT + np.asarray(design_parameters, dtype=np.float64).reshape(-1)
  sigmoid = 1.0 / (1.0 + np.exp(-shifted_logit))
  values = POISSON_MIN_VALUE + (POISSON_MAX_VALUE - POISSON_MIN_VALUE) * sigmoid
  chain = (values - POISSON_MIN_VALUE) * (POISSON_MAX_VALUE - values) / (POISSON_MAX_VALUE - POISSON_MIN_VALUE)
  return values, chain


def young_values_from_design(design_parameters):
  shifted_logit = YOUNG_BASE_LOGIT + np.asarray(design_parameters, dtype=np.float64).reshape(-1)
  sigmoid = 1.0 / (1.0 + np.exp(-shifted_logit))
  values = YOUNG_MIN_VALUE + (YOUNG_MAX_VALUE - YOUNG_MIN_VALUE) * sigmoid
  chain = (values - YOUNG_MIN_VALUE) * (YOUNG_MAX_VALUE - values) / (YOUNG_MAX_VALUE - YOUNG_MIN_VALUE)
  return values, chain


def material_statistics(young_values, poisson_values):
  return {
    "young": {"minimum": float(np.min(young_values)), "maximum": float(np.max(young_values)), "mean": float(np.mean(young_values)), "std": float(np.std(young_values))},
    "poisson": {"minimum": float(np.min(poisson_values)), "maximum": float(np.max(poisson_values)), "mean": float(np.mean(poisson_values)), "std": float(np.std(poisson_values))},
  }


def format_material_statistics(statistics):
  young_summary = statistics["young"]
  poisson_summary = statistics["poisson"]
  return f"young mean={young_summary['mean']:.6g} [{young_summary['minimum']:.6g}, {young_summary['maximum']:.6g}], poisson mean={poisson_summary['mean']:.6g} [{poisson_summary['minimum']:.6g}, {poisson_summary['maximum']:.6g}]"


def gpu_memory_used():
  free_bytes, total_bytes = cuda.mem_get_info()
  return total_bytes - free_bytes, total_bytes


def gpu_memory_summary(start_used, peak_used, end_used, total_bytes):
  bytes_per_gib = 1024.0 ** 3
  execution_baseline = GPU_EXECUTION_BASELINE_USED_BYTES
  return {"start_used_gib": start_used / bytes_per_gib, "peak_used_gib": peak_used / bytes_per_gib, "end_used_gib": end_used / bytes_per_gib, "total_gib": total_bytes / bytes_per_gib, "system_baseline_used_gib": execution_baseline / bytes_per_gib, "start_above_system_baseline_gib": max(0, start_used - execution_baseline) / bytes_per_gib, "peak_above_system_baseline_gib": max(0, peak_used - execution_baseline) / bytes_per_gib, "end_above_system_baseline_gib": max(0, end_used - execution_baseline) / bytes_per_gib, "phase_peak_growth_gib": max(0, peak_used - start_used) / bytes_per_gib}


CUDA_CONTEXT_BASELINE_USED_BYTES, CUDA_CONTEXT_TOTAL_BYTES = gpu_memory_used()
GPU_EXECUTION_BASELINE_USED_BYTES = SYSTEM_GPU_BASELINE_USED_BYTES if SYSTEM_GPU_BASELINE_USED_BYTES is not None else CUDA_CONTEXT_BASELINE_USED_BYTES
bytes_per_gib = 1024.0 ** 3
print(f"GPU memory baseline before CUDA imports={GPU_EXECUTION_BASELINE_USED_BYTES / bytes_per_gib:.6f} GiB ({'nvidia-smi whole-device reading' if SYSTEM_GPU_BASELINE_USED_BYTES is not None else 'CUDA-context fallback'}); after CUDA/YASPS imports before scene={CUDA_CONTEXT_BASELINE_USED_BYTES / bytes_per_gib:.6f} GiB.")


def maximum_absolute_value(values):
  return float(np.max(np.abs(np.asarray(values, dtype=np.float64)), initial=0.0))


##################################################################
## Load matching rest and pinched tetrahedral meshes
##################################################################
rest_grid = pv.read(REST_MESH_PATH)
pinched_grid = pv.read(PINCHED_MESH_PATH)
if pv.CellType.TETRA not in rest_grid.cells_dict or pv.CellType.TETRA not in pinched_grid.cells_dict:
  raise ValueError("Both input meshes must contain linear tetrahedra.")

rest_position_values = np.asarray(rest_grid.points, dtype=np.float64) * 1e-3
pinched_position_values = np.asarray(pinched_grid.points, dtype=np.float64) * 1e-3
tetrahedron_indices = np.asarray(rest_grid.cells_dict[pv.CellType.TETRA], dtype=np.uint32)
pinched_tetrahedron_indices = np.asarray(pinched_grid.cells_dict[pv.CellType.TETRA], dtype=np.uint32)
if rest_position_values.shape != pinched_position_values.shape:
  raise ValueError("The rest and pinched meshes have different vertex counts.")
if not np.array_equal(tetrahedron_indices, pinched_tetrahedron_indices):
  raise ValueError("The rest and pinched meshes do not have matching tetrahedra.")

controlled_indices = np.flatnonzero(rest_position_values[:, 0] < CONTROL_X_MAX).astype(np.uint32)
target_indices = np.flatnonzero(rest_position_values[:, 0] > TARGET_X_MIN).astype(np.uint32)
if controlled_indices.size == 0 or target_indices.size == 0:
  raise ValueError("The mesh must contain vertices on both sides of the clamp thresholds.")
if np.intersect1d(controlled_indices, target_indices).size != 0:
  raise ValueError("The controlled and observed target clamps overlap.")
controlled_rest_positions = rest_position_values[controlled_indices]
controlled_target_positions = pinched_position_values[controlled_indices]
response_target_positions = pinched_position_values[target_indices]


##################################################################
## Compute fixed, rest-volume-lumped vertex masses
##################################################################
tetrahedron_positions = rest_position_values[tetrahedron_indices]
edge_matrices = np.stack([tetrahedron_positions[:, 1] - tetrahedron_positions[:, 0], tetrahedron_positions[:, 2] - tetrahedron_positions[:, 0], tetrahedron_positions[:, 3] - tetrahedron_positions[:, 0]], axis=2)
tetrahedron_volumes = np.abs(np.linalg.det(edge_matrices)) / 6.0
if np.any(tetrahedron_volumes <= 0.0):
  raise ValueError("The rest mesh contains a degenerate tetrahedron.")
vertex_masses = np.zeros(rest_position_values.shape[0], dtype=np.float64)
np.add.at(vertex_masses, tetrahedron_indices.reshape(-1), np.repeat(0.25 * DENSITY_VALUE * tetrahedron_volumes, 4))
total_volume = float(tetrahedron_volumes.sum())
total_mass = float(vertex_masses.sum())
num_vertices = rest_position_values.shape[0]
num_tetrahedra = tetrahedron_indices.shape[0]

print(f"Loaded {num_vertices} vertices and {num_tetrahedra} tetrahedra.")
print(f"Driving {controlled_indices.size} vertices with x < {CONTROL_X_MAX:.2f} m and observing {target_indices.size} vertices with x > {TARGET_X_MIN:.2f} m.")
print(f"Material design: one Young modulus and Poisson ratio per tetrahedron, initialized at E={YOUNG_VALUE:.1f} Pa and nu={POISSON_VALUE:.3f}.")
print(f"Rest volume={total_volume:.8e} m^3, total mass={total_mass:.8e} kg, gravity=0.")


##################################################################
## Construct the YASPS scene and per-tetrahedron material fields
##################################################################
simulation = scene("double_clamp_per_tetrahedron_material_inverse")
dt = simulation.addConstant("dt", rows=1, cols=1)
dt.updateValue([DT_VALUE])

double_clamp = simulation.addMesh("double_clamp")
vertices = double_clamp.addPrimitive("vertices", numInstances=num_vertices)
position = vertices.addAttribute("position", rows=3, cols=1)
rest_position = vertices.addConstant("rest_position", rows=3, cols=1)
last_position = vertices.addConstant("last_position", rows=3, cols=1)
last_last_position = vertices.addConstant("last_last_position", rows=3, cols=1)
mass = vertices.addConstant("mass", rows=1, cols=1)
velocity = (last_position - last_last_position) / dt
position.updateValue(rest_position_values.reshape(-1))
rest_position.updateValue(rest_position_values.reshape(-1))
last_position.updateValue(rest_position_values.reshape(-1))
last_last_position.updateValue(rest_position_values.reshape(-1))
mass.updateValue(vertex_masses)

tetrahedra = double_clamp.addPrimitive("tetrahedra", numInstances=num_tetrahedra)
tetrahedra_to_vertices = tetrahedra.addConnectivity("tetrahedra_to_vertices", vertices, tetrahedron_indices, 4)
tetrahedron_position = tetrahedra.addAttribute("positions", through=tetrahedra_to_vertices, source=position)
tetrahedron_rest_position = tetrahedra.addAttribute("rest_positions", through=tetrahedra_to_vertices, source=rest_position)
young = tetrahedra.addConstant("young", rows=1, cols=1)
poisson = tetrahedra.addConstant("poisson", rows=1, cols=1)
young.updateValue(np.full(num_tetrahedra, YOUNG_VALUE, dtype=np.float64))
poisson.updateValue(np.full(num_tetrahedra, POISSON_VALUE, dtype=np.float64))
mu_lame = young / (2.0 * (1.0 + poisson))
lambda_lame = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))

controlled_vertices = double_clamp.addPrimitive("controlled_vertices", numInstances=controlled_indices.size)
controlled_to_vertices = controlled_vertices.addConnectivity("controlled_to_vertices", vertices, controlled_indices.reshape((-1, 1)), 1)
controlled_position = controlled_vertices.addAttribute("position", through=controlled_to_vertices, source=position)
controlled_target_position = controlled_vertices.addConstant("target_position", rows=1, cols=3)
controlled_target_position.updateValue(controlled_rest_positions.reshape(-1))
controlled_stiffness = controlled_vertices.addConstant("target_stiffness", rows=1, cols=1)
controlled_stiffness.updateValue(np.full(controlled_indices.size, PIN_STIFFNESS_VALUE, dtype=np.float64))

target_vertices = double_clamp.addPrimitive("target_vertices", numInstances=target_indices.size)
target_to_vertices = target_vertices.addConnectivity("target_to_vertices", vertices, target_indices.reshape((-1, 1)), 1)
response_position = target_vertices.addAttribute("position", through=target_to_vertices, source=position)
response_target_position = target_vertices.addConstant("target_position", rows=1, cols=3)
response_target_position.updateValue(response_target_positions.reshape(-1))
response_loss_weight = target_vertices.addConstant("loss_weight", rows=1, cols=1)
response_loss_weight.updateValue(np.full(target_indices.size, TARGET_LOSS_WEIGHT_VALUE, dtype=np.float64))
build_backward_operators = args.adjoint_steps > 0 or args.gradient_check or args.resume_checkpoint is not None


##################################################################
## Define forward energies and the final right-clamp loss
##################################################################
elastic_energy = tetrahedra.addAttribute("stable_neo_hookean_energy", computed_attribute=stable_neo_hookean(tetrahedron_rest_position, tetrahedron_position, mu_lame, lambda_lame, dt))
inertia_energy = vertices.addAttribute("zero_gravity_inertia_energy", computed_attribute=inertia(last_position, velocity, dt, position, mass))
controlled_energy = controlled_vertices.addAttribute("left_clamp_target_energy", computed_attribute=moving_energy(controlled_position, controlled_target_position, dt, controlled_stiffness))
response_difference = response_position - response_target_position
response_target_loss = target_vertices.addAttribute("right_clamp_target_loss", computed_attribute=0.5 * response_loss_weight * response_difference.dot(response_difference))
response_state_loss_gradient = differentiator().diff1([response_target_loss], [position]) if build_backward_operators else None

simulation.addEnergy(elastic_energy, projection_method=1)
simulation.addEnergy(inertia_energy, projection_method=-1)
simulation.addEnergy(controlled_energy, projection_method=-1)
simulation.addMinimizeTarget([position])


##################################################################
## Prebuild the projected adjoint Hessian and residual Jacobians
##################################################################
adjoint_hessian = None
previous_position_jacobian = None
previous_previous_position_jacobian = None
young_jacobian = None
poisson_jacobian = None
adjoint_linear_solver = None
adjoint_initial_guess = None
if build_backward_operators:
  adjoint_hessian = differentiator().diff2([elastic_energy], [position], [position], projection_method=1)
  adjoint_hessian = adjoint_hessian + differentiator().diff2([inertia_energy], [position], [position], projection_method=-1)
  adjoint_hessian = adjoint_hessian + differentiator().diff2([controlled_energy], [position], [position], projection_method=-1)
  previous_position_jacobian = differentiator().diff2([inertia_energy], [position], [last_position])
  previous_previous_position_jacobian = differentiator().diff2([inertia_energy], [position], [last_last_position])
  young_jacobian = differentiator().diff2([elastic_energy], [position], [young])
  poisson_jacobian = differentiator().diff2([elastic_energy], [position], [poisson])
  adjoint_linear_solver = solver()
  adjoint_initial_guess = gpuarray.zeros(position.size, np.float64)


def solve_adjoint_system(right_hand_side):
  hessian_start = time.time()
  adjoint_hessian.compute()
  hessian_seconds = time.time() - hessian_start
  solve_start = time.time()
  error_code = adjoint_linear_solver.computeSolution(adjoint_hessian, right_hand_side, adjoint_initial_guess, tolerance=ADJOINT_SOLVER_TOLERANCE, maxIterations=MAX_CG_ITERATIONS, zero_initial_guess=True)
  solve_seconds = time.time() - solve_start
  if error_code < 0:
    raise RuntimeError(f"Projected adjoint GPU CG failed with error code {error_code}.")
  result = vector(right_hand_side.size)
  result.updateValue(adjoint_linear_solver.solution)
  return result, hessian_seconds, solve_seconds


##################################################################
## Construct the rendered surface and target markers
##################################################################
surface_triangles = extract_surface_triangles(tetrahedron_indices)
surface_indices = np.unique(surface_triangles).astype(np.uint32)
surface_local_indices = np.full(num_vertices, -1, dtype=np.int64)
surface_local_indices[surface_indices] = np.arange(surface_indices.size)
surface_triangles_local = surface_local_indices[surface_triangles]
if np.any(surface_triangles_local < 0):
  raise RuntimeError("A surface triangle contains a non-surface vertex.")
surface_cells = np.hstack([np.full((surface_triangles.shape[0], 1), 3, dtype=np.uint32), surface_triangles_local.astype(np.uint32)])
surface_poly = pv.PolyData(rest_position_values[surface_indices], surface_cells)
controlled_surface_indices = np.intersect1d(surface_indices, controlled_indices, assume_unique=True)
target_surface_indices = np.intersect1d(surface_indices, target_indices, assume_unique=True)
controlled_target_points = pv.PolyData(pinched_position_values[controlled_surface_indices])
response_target_points = pv.PolyData(pinched_position_values[target_surface_indices])

plotter = pv.Plotter(window_size=[1920, 1080], off_screen=args.no_gui)
plotter.set_background("#f4f5f2")
plotter.add_mesh(surface_poly, color="#6aa879", opacity=0.82, smooth_shading=True, show_edges=False, specular=0.2)
plotter.add_points(controlled_target_points, color="#d34b3f", point_size=5, render_points_as_spheres=True)
plotter.add_points(response_target_points, color="#3465c5", point_size=5, render_points_as_spheres=True)
plotter.add_text("Red: driven left clamp   Blue: target right clamp", name="legend", position="lower_left", font_size=14, color="#202124")
plotter.add_text("Double-clamp inverse design", name="frame_status", position="upper_left", font_size=16, color="#202124")
plotter.camera_position = [(0.0, 0.0, 0.30), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
plotter.enable_parallel_projection()
plotter.camera.parallel_scale = 0.034

if args.preview_only:
  if args.no_gui:
    preview_path = os.path.join(SCRIPT_DIR, "preview.png")
    plotter.show(auto_close=False)
    plotter.screenshot(preview_path)
    plotter.close()
    print(f"Saved preview: {preview_path}")
  else:
    plotter.show(auto_close=True)
  raise SystemExit(0)

if args.no_gui:
  plotter.show(auto_close=False)
else:
  plotter.show(interactive_update=True, auto_close=False)

output_directory = os.path.abspath(args.output_directory or os.path.join(SCRIPT_DIR, "outputs", "inverse_per_tet_adam"))
os.makedirs(output_directory, exist_ok=True)


def output_paths(adjoint_round):
  round_name = "baseline" if adjoint_round == 0 else f"adjoint_{adjoint_round:02d}"
  round_directory = os.path.join(output_directory, round_name)
  return round_name, os.path.join(round_directory, "frames"), os.path.join(output_directory, f"{round_name}.mp4")


def encode_saved_frames(adjoint_round, loss, rms_distance, baseline_loss, baseline_rms_distance):
  round_name, frame_directory, video_path = output_paths(adjoint_round)
  ffmpeg = shutil.which("ffmpeg")
  if ffmpeg is None:
    raise RuntimeError("--save-frames requires ffmpeg.")
  loss_reduction_percent = 100.0 * (baseline_loss - loss) / max(abs(baseline_loss), 1e-30)
  rms_improvement_percent = 100.0 * (baseline_rms_distance - rms_distance) / max(abs(baseline_rms_distance), 1e-30)
  text = f"{'unoptimized baseline' if adjoint_round == 0 else f'adjoint {adjoint_round:04d}/{args.adjoint_steps:04d}'}   loss {loss:.8f}   loss reduction {loss_reduction_percent:.2f}%   RMS improvement {rms_improvement_percent:.2f}%"
  command = [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(args.video_fps), "-start_number", "0", "-i", os.path.join(frame_directory, "frame_%04d.png"), "-frames:v", str(args.num_frames), "-vf", f"drawtext=text='{text}':expansion=none:fontcolor=black:fontsize=36:box=1:boxcolor=white@0.78:boxborderw=12:x=40:y=h-th-40", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", video_path]
  subprocess.run(command, check=True)
  print(f"Saved {round_name} video: {video_path}")
  return video_path


def refresh_surface(render):
  current_positions = position.compute().value.get().reshape((-1, 3))
  surface_poly.points = current_positions[surface_indices]
  if render:
    surface_poly.compute_normals(inplace=True)
    plotter.render()
    if not args.no_gui:
      plotter.update()
  return current_positions


##################################################################
## Run one forward trajectory and retain converged states
##################################################################
def run_forward(young_design, poisson_design, adjoint_round=None, baseline_loss=None, baseline_rms_distance=None):
  gpu_start_used, gpu_total = gpu_memory_used()
  gpu_peak_used = gpu_start_used
  young_values, young_chain = young_values_from_design(young_design)
  poisson_values, poisson_chain = poisson_values_from_design(poisson_design)
  young.updateValue(young_values)
  poisson.updateValue(poisson_values)
  material_summary = material_statistics(young_values, poisson_values)
  position.updateValue(rest_position_values.reshape(-1))
  last_position.updateValue(rest_position_values.reshape(-1))
  last_last_position.updateValue(rest_position_values.reshape(-1))
  controlled_target_position.updateValue(controlled_rest_positions.reshape(-1))

  save_this_forward = adjoint_round is not None and (adjoint_round == 0 or adjoint_round % args.video_every == 0 or adjoint_round == args.adjoint_steps)
  render_this_forward = (save_this_forward and args.save_frames) or not args.no_gui
  saved_positions = [rest_position_values.copy()]
  saved_control_targets = []
  maximum_pin_errors = []
  response_rms_history = []
  saved_video = None
  round_name = "reconstruction" if adjoint_round is None else output_paths(adjoint_round)[0]
  frame_directory = output_paths(adjoint_round)[1] if save_this_forward else None
  if adjoint_round is not None:
    plotter.add_text(f"{'Unoptimized baseline' if adjoint_round == 0 else f'After Adam {adjoint_round:02d}/{args.adjoint_steps:02d}'}\n{format_material_statistics(material_summary)}", name="frame_status", position="upper_left", font_size=15, color="#202124")
    if save_this_forward and args.save_frames:
      if os.path.isdir(frame_directory):
        shutil.rmtree(frame_directory)
      os.makedirs(frame_directory, exist_ok=True)

  forward_start = time.time()
  for frame in range(args.num_frames):
    ramp_phase = (frame + 1) / args.num_frames
    pin_progress = 0.5 - 0.5 * np.cos(np.pi * ramp_phase)
    active_targets = controlled_rest_positions + pin_progress * (controlled_target_positions - controlled_rest_positions)
    controlled_target_position.updateValue(active_targets.reshape(-1))
    last_last_position.updateValue(last_position.value, deepCopy=True)
    last_position.updateValue(position.value, deepCopy=True)

    converged = False
    frame_start = time.time()
    for newton_iteration in range(MAX_NEWTON_ITERATIONS):
      energy_before = simulation.computeTotalEnergy()
      position_before = position.value.copy()
      solve_start = time.time()
      displacement = simulation.minimizeEnergy(tolerance=SOLVER_TOLERANCE, maxIterations=MAX_CG_ITERATIONS)[0]
      solve_seconds = time.time() - solve_start
      max_displacement = float(gpuarray.max(abs(displacement)).get())
      gpu_current_used, _ = gpu_memory_used()
      gpu_peak_used = max(gpu_peak_used, gpu_current_used)

      step_taken = 1.0
      accepted = False
      energy_after = energy_before
      for _ in range(MAX_LINE_SEARCH_STEPS):
        position.updateValue(position_before - step_taken * displacement, deepCopy=True)
        energy_after = simulation.computeTotalEnergy()
        if np.isfinite(energy_after) and energy_after <= energy_before + 1e-12 * max(1.0, abs(energy_before)):
          accepted = True
          break
        step_taken *= 0.5
      if not accepted:
        position.updateValue(position_before, deepCopy=True)
        if max_displacement > 10.0 * MOTION_TOLERANCE:
          raise RuntimeError(f"Newton line search failed at frame {frame}, iteration {newton_iteration}.")
        step_taken = 0.0
        energy_after = energy_before

      print(f"{round_name} frame={frame:03d} newton={newton_iteration:03d} solver={solve_seconds:.4f}s step={step_taken:.6f} max_dx={max_displacement:.6e} energy={energy_before:.8e}->{energy_after:.8e}")
      if step_taken * max_displacement < MOTION_TOLERANCE:
        converged = True
        break
    if not converged:
      print(f"Warning: {round_name} frame {frame} reached the {MAX_NEWTON_ITERATIONS}-iteration Newton cap.")

    current_positions = refresh_surface(render_this_forward)
    pin_error = float(np.linalg.norm(current_positions[controlled_indices] - active_targets, axis=1).max())
    response_distances = np.linalg.norm(current_positions[target_indices] - response_target_positions, axis=1)
    response_rms = float(np.sqrt(np.mean(response_distances * response_distances)))
    maximum_pin_errors.append(pin_error)
    response_rms_history.append(response_rms)
    saved_positions.append(current_positions.copy())
    saved_control_targets.append(active_targets.copy())
    if save_this_forward and args.save_frames:
      plotter.screenshot(os.path.join(frame_directory, f"frame_{frame:04d}.png"))
    print(f"completed {round_name} frame={frame:03d} seconds={time.time() - frame_start:.3f} pin_progress={pin_progress:.6f} max_pin_error={pin_error:.6e} right_rms={response_rms:.6e}")

  loss = float(response_target_loss.compute().value.get().sum())
  final_positions = saved_positions[-1]
  final_distances = np.linalg.norm(final_positions[target_indices] - response_target_positions, axis=1)
  terminal_summary = {
    "rms_distance": float(np.sqrt(np.mean(final_distances * final_distances))),
    "mean_distance": float(np.mean(final_distances)),
    "maximum_distance": float(np.max(final_distances)),
    "minimum_distance": float(np.min(final_distances)),
    "controlled_maximum_error": maximum_pin_errors[-1],
  }
  elapsed = time.time() - forward_start
  gpu_end_used, _ = gpu_memory_used()
  gpu_peak_used = max(gpu_peak_used, gpu_end_used)
  memory = gpu_memory_summary(gpu_start_used, gpu_peak_used, gpu_end_used, gpu_total)
  print(f"Finished forward {round_name} in {elapsed:.2f}s: loss={loss:.8e}, right_clamp={terminal_summary}, maximum_pin_error={max(maximum_pin_errors):.8e}, {format_material_statistics(material_summary)}, gpu_memory={memory}.")
  if save_this_forward and args.save_frames:
    reference_loss = loss if adjoint_round == 0 else baseline_loss
    reference_rms = terminal_summary["rms_distance"] if adjoint_round == 0 else baseline_rms_distance
    saved_video = encode_saved_frames(adjoint_round, loss, terminal_summary["rms_distance"], reference_loss, reference_rms)
  return {
    "positions": saved_positions,
    "control_targets": saved_control_targets,
    "loss": loss,
    "terminal_summary": terminal_summary,
    "response_rms_history": response_rms_history,
    "maximum_pin_error": max(maximum_pin_errors),
    "elapsed_seconds": elapsed,
    "gpu_memory": memory,
    "video": saved_video,
    "young_values": young_values,
    "young_chain": young_chain,
    "poisson_values": poisson_values,
    "poisson_chain": poisson_chain,
    "material_summary": material_summary,
  }


##################################################################
## Replay the converged trajectory and accumulate material gradients
##################################################################
def run_backward(trajectory, adjoint_round):
  gpu_start_used, gpu_total = gpu_memory_used()
  gpu_peak_used = gpu_start_used
  position_adjoints = [vector(response_state_loss_gradient.size) for _ in range(args.num_frames + 2)]
  young_gradient = vector(young_jacobian.cols)
  poisson_gradient = vector(poisson_jacobian.cols)
  gpu_current_used, _ = gpu_memory_used()
  gpu_peak_used = max(gpu_peak_used, gpu_current_used)
  backward_start = time.time()
  timing = {"state_restore_seconds": 0.0, "loss_gradient_seconds": 0.0, "projected_hessian_seconds": 0.0, "gpu_cg_seconds": 0.0, "jacobian_and_transpose_spmv_seconds": 0.0}

  for frame in range(args.num_frames - 1, -1, -1):
    restore_start = time.time()
    last_position.updateValue(trajectory["positions"][frame].reshape(-1))
    last_last_position.updateValue(trajectory["positions"][max(0, frame - 1)].reshape(-1))
    position.updateValue(trajectory["positions"][frame + 1].reshape(-1))
    controlled_target_position.updateValue(trajectory["control_targets"][frame].reshape(-1))
    timing["state_restore_seconds"] += time.time() - restore_start

    loss_gradient_start = time.time()
    current_loss_gradient_inf = 0.0
    if frame == args.num_frames - 1:
      response_state_loss_gradient.compute()
      current_loss_gradient = vector(response_state_loss_gradient.size)
      current_loss_gradient.updateValue(response_state_loss_gradient)
      current_loss_gradient_inf = float(gpuarray.max(abs(current_loss_gradient.value)).get())
      position_adjoints[frame + 2] = position_adjoints[frame + 2] + current_loss_gradient
    timing["loss_gradient_seconds"] += time.time() - loss_gradient_start

    auxiliary, hessian_seconds, solve_seconds = solve_adjoint_system(position_adjoints[frame + 2])
    timing["projected_hessian_seconds"] += hessian_seconds
    timing["gpu_cg_seconds"] += solve_seconds

    jacobian_start = time.time()
    previous_position_jacobian.compute()
    previous_position_contribution = previous_position_jacobian.spmv(auxiliary, transpose=True)
    previous_previous_position_jacobian.compute()
    previous_previous_position_contribution = previous_previous_position_jacobian.spmv(auxiliary, transpose=True)
    young_jacobian.compute()
    young_gradient = young_gradient - young_jacobian.spmv(auxiliary, transpose=True)
    poisson_jacobian.compute()
    poisson_gradient = poisson_gradient - poisson_jacobian.spmv(auxiliary, transpose=True)
    timing["jacobian_and_transpose_spmv_seconds"] += time.time() - jacobian_start

    position_adjoints[frame + 1] = position_adjoints[frame + 1] - previous_position_contribution
    position_adjoints[frame] = position_adjoints[frame] - previous_previous_position_contribution
    gpu_current_used, _ = gpu_memory_used()
    gpu_peak_used = max(gpu_peak_used, gpu_current_used)
    if frame == args.num_frames - 1 or frame == 0 or frame % max(1, args.num_frames // 10) == 0:
      print(f"adjoint={adjoint_round:02d} reverse_frame={frame:03d} |g_current|_inf={current_loss_gradient_inf:.6e} |lambda_x|_inf={float(gpuarray.max(abs(position_adjoints[frame + 1].value)).get()):.6e} solver=gpu_cg")

  young_gradient_values = young_gradient.value.get()
  poisson_gradient_values = poisson_gradient.value.get()
  elapsed = time.time() - backward_start
  gpu_end_used, _ = gpu_memory_used()
  gpu_peak_used = max(gpu_peak_used, gpu_end_used)
  memory = gpu_memory_summary(gpu_start_used, gpu_peak_used, gpu_end_used, gpu_total)
  timing["other_seconds"] = elapsed - sum(timing.values())
  print(f"Finished adjoint {adjoint_round:02d} in {elapsed:.2f}s: young_gradient l2={np.linalg.norm(young_gradient_values):.6e} inf={maximum_absolute_value(young_gradient_values):.6e}, poisson_gradient l2={np.linalg.norm(poisson_gradient_values):.6e} inf={maximum_absolute_value(poisson_gradient_values):.6e}, timing={timing}, gpu_memory={memory}.")
  return young_gradient_values, poisson_gradient_values, elapsed, timing, memory


##################################################################
## Initialize or resume Adam, then alternate backward and forward
##################################################################
optimization_start = time.time()
optimization_history = []
resume_completed_round = 0
resumed_from_checkpoint = None
baseline_forward_gpu_memory = None

if args.resume_checkpoint is None:
  young_design = np.zeros(num_tetrahedra, dtype=np.float64)
  poisson_design = np.zeros(num_tetrahedra, dtype=np.float64)
  young_first_moment = np.zeros_like(young_design)
  young_second_moment = np.zeros_like(young_design)
  poisson_first_moment = np.zeros_like(poisson_design)
  poisson_second_moment = np.zeros_like(poisson_design)
  baseline_trajectory = run_forward(young_design, poisson_design, adjoint_round=0)
  baseline_forward_gpu_memory = baseline_trajectory["gpu_memory"]
  trajectory = baseline_trajectory
  initial_loss = trajectory["loss"]
  initial_rms_distance = trajectory["terminal_summary"]["rms_distance"]
  best_loss = initial_loss
  best_rms_distance = initial_rms_distance
  best_round = 0
  best_young_design = young_design.copy()
  best_poisson_design = poisson_design.copy()
else:
  resumed_from_checkpoint = os.path.abspath(args.resume_checkpoint)
  with np.load(resumed_from_checkpoint, allow_pickle=False) as checkpoint:
    resume_completed_round = int(checkpoint["completed_adjoint_round"].item())
    initial_loss = float(checkpoint["initial_loss"].item())
    initial_rms_distance = float(checkpoint["initial_rms_distance"].item())
    young_design = checkpoint["young_design"].copy()
    poisson_design = checkpoint["poisson_design"].copy()
    young_first_moment = checkpoint["young_first_moment"].copy()
    young_second_moment = checkpoint["young_second_moment"].copy()
    poisson_first_moment = checkpoint["poisson_first_moment"].copy()
    poisson_second_moment = checkpoint["poisson_second_moment"].copy()
    best_loss = float(checkpoint["best_loss"].item())
    best_rms_distance = float(checkpoint["best_rms_distance"].item())
    best_round = int(checkpoint["best_adjoint_round"].item())
    best_young_design = checkpoint["best_young_design"].copy()
    best_poisson_design = checkpoint["best_poisson_design"].copy()
  if resume_completed_round >= args.adjoint_steps:
    raise ValueError("The checkpoint has already completed the requested number of adjoint rounds.")
  print(f"Reconstructing trajectory from completed Adam round {resume_completed_round:02d}.")
  trajectory = run_forward(young_design, poisson_design)

if args.gradient_check:
  young_value_gradient, poisson_value_gradient, _, _, _ = run_backward(trajectory, 1)
  young_design_gradient = young_value_gradient * trajectory["young_chain"]
  poisson_design_gradient = poisson_value_gradient * trajectory["poisson_chain"]
  epsilon = 1e-1
  checks = []
  for name, design, other_design, gradient in [("young", young_design, poisson_design, young_design_gradient), ("poisson", poisson_design, young_design, poisson_design_gradient)]:
    index = int(np.argmax(np.abs(gradient)))
    plus = design.copy()
    minus = design.copy()
    plus[index] += epsilon
    minus[index] -= epsilon
    if name == "young":
      plus_loss = run_forward(plus, other_design)["loss"]
      minus_loss = run_forward(minus, other_design)["loss"]
    else:
      plus_loss = run_forward(other_design, plus)["loss"]
      minus_loss = run_forward(other_design, minus)["loss"]
    finite_difference = (plus_loss - minus_loss) / (2.0 * epsilon)
    adjoint_value = float(gradient[index])
    checks.append({"parameter": name, "tetrahedron": index, "adjoint": adjoint_value, "finite_difference": finite_difference, "relative_error": abs(adjoint_value - finite_difference) / max(1e-30, abs(adjoint_value), abs(finite_difference)), "same_sign": bool(adjoint_value * finite_difference > 0.0)})
  print(f"Gradient check: {json.dumps(checks, indent=2)}")
  plotter.close()
  raise SystemExit(0)

for adjoint_round in range(resume_completed_round + 1, args.adjoint_steps + 1):
  young_value_gradient, poisson_value_gradient, backward_seconds, backward_timing, backward_memory = run_backward(trajectory, adjoint_round)
  young_gradient = young_value_gradient * trajectory["young_chain"]
  poisson_gradient = poisson_value_gradient * trajectory["poisson_chain"]
  if not np.all(np.isfinite(young_gradient)) or not np.all(np.isfinite(poisson_gradient)):
    raise RuntimeError("The material design gradient contains a non-finite value.")
  if maximum_absolute_value(young_gradient) == 0.0 and maximum_absolute_value(poisson_gradient) == 0.0:
    raise RuntimeError("Both material design gradients are zero.")

  previous_loss = trajectory["loss"]
  previous_rms_distance = trajectory["terminal_summary"]["rms_distance"]
  young_first_moment = ADAM_BETA1 * young_first_moment + (1.0 - ADAM_BETA1) * young_gradient
  young_second_moment = ADAM_BETA2 * young_second_moment + (1.0 - ADAM_BETA2) * young_gradient * young_gradient
  poisson_first_moment = ADAM_BETA1 * poisson_first_moment + (1.0 - ADAM_BETA1) * poisson_gradient
  poisson_second_moment = ADAM_BETA2 * poisson_second_moment + (1.0 - ADAM_BETA2) * poisson_gradient * poisson_gradient
  young_corrected_first = young_first_moment / (1.0 - ADAM_BETA1 ** adjoint_round)
  young_corrected_second = young_second_moment / (1.0 - ADAM_BETA2 ** adjoint_round)
  poisson_corrected_first = poisson_first_moment / (1.0 - ADAM_BETA1 ** adjoint_round)
  poisson_corrected_second = poisson_second_moment / (1.0 - ADAM_BETA2 ** adjoint_round)
  raw_young_step = -args.adam_learning_rate * young_corrected_first / (np.sqrt(young_corrected_second) + ADAM_EPSILON)
  raw_poisson_step = -args.adam_learning_rate * poisson_corrected_first / (np.sqrt(poisson_corrected_second) + ADAM_EPSILON)
  young_step_scale = min(1.0, args.young_step_size / max(maximum_absolute_value(raw_young_step), 1e-30))
  poisson_step_scale = min(1.0, args.poisson_step_size / max(maximum_absolute_value(raw_poisson_step), 1e-30))
  young_step = young_step_scale * raw_young_step
  poisson_step = poisson_step_scale * raw_poisson_step
  young_design = np.clip(young_design + young_step, -YOUNG_LOGIT_DESIGN_LIMIT, YOUNG_LOGIT_DESIGN_LIMIT)
  poisson_design = np.clip(poisson_design + poisson_step, -POISSON_LOGIT_DESIGN_LIMIT, POISSON_LOGIT_DESIGN_LIMIT)

  trajectory = run_forward(young_design, poisson_design, adjoint_round=adjoint_round, baseline_loss=initial_loss, baseline_rms_distance=initial_rms_distance)
  current_loss = trajectory["loss"]
  current_rms_distance = trajectory["terminal_summary"]["rms_distance"]
  loss_reduction = (initial_loss - current_loss) / max(initial_loss, 1e-30)
  rms_reduction = (initial_rms_distance - current_rms_distance) / max(initial_rms_distance, 1e-30)
  record = {
    "adjoint_round": adjoint_round,
    "loss_before": previous_loss,
    "loss_after": current_loss,
    "loss_decreased": bool(current_loss < previous_loss),
    "relative_loss_reduction_from_baseline": loss_reduction,
    "rms_distance_before": previous_rms_distance,
    "rms_distance_after": current_rms_distance,
    "relative_rms_reduction_from_baseline": rms_reduction,
    "adam_learning_rate": args.adam_learning_rate,
    "maximum_young_design_step": maximum_absolute_value(young_step),
    "maximum_poisson_design_step": maximum_absolute_value(poisson_step),
    "forward_seconds": trajectory["elapsed_seconds"],
    "forward_gpu_memory": trajectory["gpu_memory"],
    "backward_seconds": backward_seconds,
    "backward_timing": backward_timing,
    "backward_gpu_memory": backward_memory,
    "maximum_pin_error": trajectory["maximum_pin_error"],
    "terminal_summary": trajectory["terminal_summary"],
    "material_statistics": trajectory["material_summary"],
    "gradient_statistics": {
      "young_design_l2": float(np.linalg.norm(young_gradient)),
      "young_design_inf": maximum_absolute_value(young_gradient),
      "poisson_design_l2": float(np.linalg.norm(poisson_gradient)),
      "poisson_design_inf": maximum_absolute_value(poisson_gradient),
    },
    "video": trajectory["video"],
  }
  optimization_history.append(record)
  if current_loss < best_loss:
    best_loss = current_loss
    best_rms_distance = current_rms_distance
    best_round = adjoint_round
    best_young_design = young_design.copy()
    best_poisson_design = poisson_design.copy()

  checkpoint_path = os.path.join(output_directory, "latest_checkpoint.npz")
  temporary_checkpoint_path = os.path.join(output_directory, "latest_checkpoint.tmp.npz")
  np.savez(temporary_checkpoint_path, completed_adjoint_round=adjoint_round, initial_loss=initial_loss, initial_rms_distance=initial_rms_distance, current_loss=current_loss, current_rms_distance=current_rms_distance, young_design=young_design, poisson_design=poisson_design, young_first_moment=young_first_moment, young_second_moment=young_second_moment, poisson_first_moment=poisson_first_moment, poisson_second_moment=poisson_second_moment, best_loss=best_loss, best_rms_distance=best_rms_distance, best_adjoint_round=best_round, best_young_design=best_young_design, best_poisson_design=best_poisson_design)
  os.replace(temporary_checkpoint_path, checkpoint_path)
  archived_checkpoint_path = None
  if adjoint_round % args.checkpoint_every == 0 or adjoint_round == args.adjoint_steps:
    archived_checkpoint_path = os.path.join(output_directory, f"checkpoint_round_{adjoint_round:04d}.npz")
    temporary_archived_checkpoint_path = os.path.join(output_directory, f"checkpoint_round_{adjoint_round:04d}.tmp.npz")
    shutil.copyfile(checkpoint_path, temporary_archived_checkpoint_path)
    os.replace(temporary_archived_checkpoint_path, archived_checkpoint_path)
    record["checkpoint"] = archived_checkpoint_path
  print(f"optimization={adjoint_round:02d}/{args.adjoint_steps:02d} loss={previous_loss:.8e}->{current_loss:.8e} loss_decreased={current_loss < previous_loss} baseline_loss_reduction={loss_reduction:.8e} right_rms={current_rms_distance:.8e} baseline_rms_reduction={rms_reduction:.8e} {format_material_statistics(trajectory['material_summary'])} young_step={maximum_absolute_value(young_step):.6e} poisson_step={maximum_absolute_value(poisson_step):.6e}")
  print(f"Saved optimization checkpoint: {checkpoint_path}")
  if archived_checkpoint_path is not None:
    print(f"Archived restart checkpoint: {archived_checkpoint_path}")


##################################################################
## Save the best/final material fields and complete run report
##################################################################
best_young_values, _ = young_values_from_design(best_young_design)
best_poisson_values, _ = poisson_values_from_design(best_poisson_design)
best_young_path = os.path.join(output_directory, "best_young.npy")
best_poisson_path = os.path.join(output_directory, "best_poisson.npy")
final_young_path = os.path.join(output_directory, "final_young.npy")
final_poisson_path = os.path.join(output_directory, "final_poisson.npy")
np.save(best_young_path, best_young_values)
np.save(best_poisson_path, best_poisson_values)
np.save(final_young_path, trajectory["young_values"])
np.save(final_poisson_path, trajectory["poisson_values"])

results = {
  "num_frames": args.num_frames,
  "dt": DT_VALUE,
  "requested_adjoint_steps": args.adjoint_steps,
  "completed_adjoint_steps": resume_completed_round + len(optimization_history),
  "resumed_from_checkpoint": resumed_from_checkpoint,
  "history_scope": "resumed segment only" if resumed_from_checkpoint is not None else "complete run",
  "optimizer": "adam",
  "optimizer_update_policy": "one backward pass and one unconditionally accepted Adam update per inverse round",
  "adam_learning_rate": args.adam_learning_rate,
  "video_every": args.video_every,
  "checkpoint_every": args.checkpoint_every,
  "adam_beta1": ADAM_BETA1,
  "adam_beta2": ADAM_BETA2,
  "loss_definition": "0.5 * weight * sum(||right_clamp_position - pinched_right_clamp_target||^2) at the final converged frame only",
  "loss_weight": TARGET_LOSS_WEIGHT_VALUE,
  "controlled_vertex_selection": f"rest x < {CONTROL_X_MAX}",
  "target_vertex_selection": f"rest x > {TARGET_X_MIN}",
  "num_controlled_vertices": int(controlled_indices.size),
  "num_target_vertices": int(target_indices.size),
  "num_vertices": num_vertices,
  "num_tetrahedra": num_tetrahedra,
  "material_layout": "one Young modulus and one Poisson ratio per tetrahedron",
  "baseline_forward_gpu_memory": baseline_forward_gpu_memory,
  "gpu_memory_measurement": "Whole-device CUDA memory sampled at phase boundaries and after every forward Newton solve or reverse frame. Application footprint subtracts the whole-device nvidia-smi reading captured before CUDA/YASPS imports; phase peak growth subtracts each phase's own starting usage.",
  "gpu_memory_baseline": {
    "system_before_cuda_import_gib": None if SYSTEM_GPU_BASELINE_USED_BYTES is None else SYSTEM_GPU_BASELINE_USED_BYTES / (1024.0 ** 3),
    "after_cuda_context_before_scene_gib": CUDA_CONTEXT_BASELINE_USED_BYTES / (1024.0 ** 3),
    "execution_baseline_used_gib": GPU_EXECUTION_BASELINE_USED_BYTES / (1024.0 ** 3),
    "cuda_reported_total_gib": CUDA_CONTEXT_TOTAL_BYTES / (1024.0 ** 3),
    "nvidia_smi_reported_total_gib": None if SYSTEM_GPU_TOTAL_BYTES is None else SYSTEM_GPU_TOTAL_BYTES / (1024.0 ** 3),
  },
  "young_bounds": [YOUNG_MIN_VALUE, YOUNG_MAX_VALUE],
  "poisson_bounds": [POISSON_MIN_VALUE, POISSON_MAX_VALUE],
  "initial_loss": initial_loss,
  "initial_rms_distance": initial_rms_distance,
  "final_loss": trajectory["loss"],
  "final_rms_distance": trajectory["terminal_summary"]["rms_distance"],
  "best_loss": best_loss,
  "best_rms_distance": best_rms_distance,
  "best_adjoint_round": best_round,
  "baseline_video": os.path.join(output_directory, "baseline.mp4") if args.save_frames else None,
  "best_video": output_paths(best_round)[2] if args.save_frames and (best_round == 0 or best_round % args.video_every == 0 or best_round == args.adjoint_steps) else None,
  "latest_checkpoint": os.path.join(output_directory, "latest_checkpoint.npz") if args.adjoint_steps > 0 else None,
  "best_young_values": best_young_path,
  "best_poisson_values": best_poisson_path,
  "final_young_values": final_young_path,
  "final_poisson_values": final_poisson_path,
  "young_jacobian": None if young_jacobian is None else {"rows": young_jacobian.rows, "cols": young_jacobian.cols, "block_dimensions": young_jacobian.block_dimensions, "block_counts": young_jacobian.block_counts},
  "poisson_jacobian": None if poisson_jacobian is None else {"rows": poisson_jacobian.rows, "cols": poisson_jacobian.cols, "block_dimensions": poisson_jacobian.block_dimensions, "block_counts": poisson_jacobian.block_counts},
  "history": optimization_history,
  "total_seconds": time.time() - optimization_start,
}
results_path = os.path.join(output_directory, "results.json")
with open(results_path, "w", encoding="utf-8") as file:
  json.dump(results, file, indent=2)

final_loss_reduction = (initial_loss - trajectory["loss"]) / max(initial_loss, 1e-30)
final_rms_reduction = (initial_rms_distance - trajectory["terminal_summary"]["rms_distance"]) / max(initial_rms_distance, 1e-30)
print(f"Inverse simulation finished in {results['total_seconds']:.2f}s: loss={initial_loss:.8e}->{trajectory['loss']:.8e}, loss_reduction={final_loss_reduction:.8e}, right_rms={initial_rms_distance:.8e}->{trajectory['terminal_summary']['rms_distance']:.8e}, rms_reduction={final_rms_reduction:.8e}, best_round={best_round}.")
print(f"Saved optimization results: {results_path}")
plotter.close()
