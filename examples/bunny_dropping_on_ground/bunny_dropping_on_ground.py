import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pyvista as pv

from yasps import attribute, differentiator, scene, solver, vector

from helpers import edge_edge, extract_edges_from_triangles, extract_surface_triangles, inertia, point_edge, point_point, point_triangle, stable_neo_hookean


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "../ccd"))
from ccd import CCD

OUTPUT_ROOT = SCRIPT_DIR

NUM_FRAMES = 200
NUM_ADJOINT_STEPS = 30
DT_VALUE = 0.01
DHAT_VALUE = 1e-6
KAPPA_VALUE = 10000.0
FLOOR_SIZE = 1000.0
FLOOR_HEIGHT = 0.0
FLOOR_CENTER = np.array([0.0, FLOOR_HEIGHT, 0.0], dtype=np.float64)
BUNNY_TOTAL_MASS = 1.0

POISSON_VALUE = 0.2645697005781997
YOUNG_VALUE = 10259.25455816859
MU_LAME_VALUE = YOUNG_VALUE / (2.0 * (1.0 + POISSON_VALUE))
LAMBDA_LAME_VALUE = (
  YOUNG_VALUE
  * POISSON_VALUE
  / ((1.0 + POISSON_VALUE) * (1.0 - 2.0 * POISSON_VALUE))
)

SOLVER_TOLERANCE = 1e-4
ADJOINT_SOLVER_TOLERANCE = 1e-8
MOTION_TOLERANCE = 1e-2
MAX_NEWTON_ITERATIONS = 10
REFRESH_MAX_NEWTON_ITERATIONS = 200
REFRESH_NEWTON_EVERY = 5
MAX_CG_ITERATIONS = 20000
MAX_LINE_SEARCH_STEPS = 8
INITIAL_POSITION_STEP_SIZE = 1.0
ADAM_LEARNING_RATE = 1.0
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8
VIDEO_FPS = round(1.0 / DT_VALUE)


# Parse runtime, rendering, optimization, and checkpoint options.
parser = argparse.ArgumentParser(description="Drop one soft tetrahedral bunny onto a large static floor.")
parser.add_argument("--num-frames", type=int, default=NUM_FRAMES)
parser.add_argument("--adjoint-steps", type=int, default=NUM_ADJOINT_STEPS, help=("Exact number of backward passes, each followed by one forward pass " f"(default: {NUM_ADJOINT_STEPS})."),)
parser.add_argument("--adam-learning-rate", type=float, default=ADAM_LEARNING_RATE, help="Adam learning rate before applying the update-length cap.",)
parser.add_argument("--initial-position-step-size", type=float, default=INITIAL_POSITION_STEP_SIZE, help=("Initial and maximum L2 length of one normalized-gradient " "initial-translation update."),)
parser.add_argument("--preview-only", action="store_true", help="Show the initial PyVista scene without running the simulation.",)
parser.add_argument("--save-frames", action="store_true", help="Save one PNG per frame and encode the frames into an MP4.",)
parser.add_argument("--save-obj", action="store_true", help="Save only the bunny surface as one OBJ per frame.",)
parser.add_argument("--video-fps", type=int, default=VIDEO_FPS, help=("MP4 frame rate used with --save-frames " f"(default: {VIDEO_FPS}, matching dt={DT_VALUE})."),)
parser.add_argument("--no-gui", action="store_true", help="Run and render off screen without opening the interactive window.",)
parser.add_argument("--output-directory", type=str, default=None, help="Override the inverse-simulation output directory.")
args = parser.parse_args()

if args.num_frames < 0:
  raise ValueError("--num-frames must be non-negative.")
if args.adjoint_steps < 0:
  raise ValueError("--adjoint-steps must be non-negative.")
if args.initial_position_step_size <= 0.0:
  raise ValueError("--initial-position-step-size must be positive.")
if args.adam_learning_rate <= 0.0:
  raise ValueError("--adam-learning-rate must be positive.")
if args.video_fps <= 0:
  raise ValueError("--video-fps must be positive.")


# Small utilities for YASPS constants, mesh construction, statistics, and memory reporting.
def add_scalar_constant(owner, name, value):
  result = owner.addConstant(name, rows=1, cols=1)
  result.updateValue([value])
  return result


def gpu_memory_used():
  free_bytes, total_bytes = cuda.mem_get_info()
  return total_bytes - free_bytes, total_bytes


def gpu_memory_summary(start_used, peak_used, end_used, total_bytes):
  bytes_per_gib = 1024.0 ** 3
  return {
    "start_used_gib": start_used / bytes_per_gib,
    "peak_used_gib": peak_used / bytes_per_gib,
    "end_used_gib": end_used / bytes_per_gib,
    "total_gib": total_bytes / bytes_per_gib,
  }


def load_bunny():
  with open(os.path.join(SCRIPT_DIR, "../data/bunny.ele"), "r") as file:
    file.readline()
    tetrahedra = [
      [int(value) - 1 for value in line.split()[3:]]
      for line in file
    ]

  with open(os.path.join(SCRIPT_DIR, "../data/bunny.node"), "r") as file:
    file.readline()
    positions = [
      [float(value) for value in line.split()[1:]]
      for line in file
    ]

  tetrahedra = np.asarray(tetrahedra, dtype=np.uint32)
  positions = np.asarray(positions, dtype=np.float64) / 5.0
  positions -= 0.5 * (positions.min(axis=0) + positions.max(axis=0))
  positions[:, 1] += 1.0
  positions[:, 0] -= 2
  positions[:, 2] -= 2.5
  return positions, tetrahedra


def make_floor():
  half_size = 0.5 * FLOOR_SIZE
  positions = np.array([[-half_size, FLOOR_HEIGHT, -half_size], [half_size, FLOOR_HEIGHT, -half_size], [half_size, FLOOR_HEIGHT, half_size], [-half_size, FLOOR_HEIGHT, half_size],], dtype=np.float64)
  triangles = np.array([[0, 2, 1], [0, 3, 2],], dtype=np.uint32)
  return positions, triangles


bunny_positions, bunny_tetrahedra = load_bunny()
bunny_surface_triangles = extract_surface_triangles(bunny_tetrahedra)
bunny_edges = extract_edges_from_triangles(bunny_surface_triangles)
bunny_surface_indices = np.unique(bunny_surface_triangles).astype(np.uint32)

floor_positions, floor_triangles = make_floor()
floor_edges = extract_edges_from_triangles(floor_triangles)
floor_surface_indices = np.arange(floor_positions.shape[0], dtype=np.uint32)

num_bunny_vertices = bunny_positions.shape[0]
num_bunny_tetrahedra = bunny_tetrahedra.shape[0]
floor_vertex_offset = num_bunny_vertices

# Build the YASPS scene and simulation state.

simulation = scene("bunny_dropping_on_ground")
dt = add_scalar_constant(simulation, "dt", DT_VALUE)
dhat = add_scalar_constant(simulation, "dhat", DHAT_VALUE)
kappa = add_scalar_constant(simulation, "kappa", KAPPA_VALUE)
initial_translation_parameter = simulation.addConstant("initial_translation", rows=3, cols=1)
initial_translation_parameter.updateValue(np.zeros(3, dtype=np.float64))

bunny = simulation.addMesh("bunny_soft")

vertices_soft = bunny.addPrimitive("vertices_soft", numInstances=num_bunny_vertices)
position = vertices_soft.addAttribute("position", rows=3, cols=1)
rest_position = vertices_soft.addConstant("rest_position", rows=3, cols=1)
last_position = vertices_soft.addConstant("last_position", rows=3, cols=1)
last_last_position = vertices_soft.addConstant("last_last_position", rows=3, cols=1)
mass = vertices_soft.addConstant("mass", rows=1, cols=1)
initial_position_mapping = vertices_soft.addAttribute("initial_position_mapping", computed_attribute=rest_position + initial_translation_parameter)
velocity = (last_position - last_last_position) / dt

position.updateValue(bunny_positions.reshape(-1))
rest_position.updateValue(bunny_positions.reshape(-1))
last_position.updateValue(bunny_positions.reshape(-1))
last_last_position.updateValue(bunny_positions.reshape(-1))
mass.updateValue(np.full(num_bunny_vertices, BUNNY_TOTAL_MASS / num_bunny_vertices, dtype=np.float64))

tets_soft = bunny.addPrimitive("tets_soft", numInstances=num_bunny_tetrahedra)
tets_to_vertices = tets_soft.addConnectivity("tets_to_vertices", vertices_soft, bunny_tetrahedra, 4)
tet_positions = tets_soft.addAttribute("positions", through=tets_to_vertices, source=position)
tet_rest_positions = tets_soft.addAttribute("rest_positions", through=tets_to_vertices, source=rest_position)
mu = add_scalar_constant(bunny, "mu", MU_LAME_VALUE)
lam = add_scalar_constant(bunny, "lambda", LAMBDA_LAME_VALUE)

floor_mesh = simulation.addMesh("floor")
floor_vertices = floor_mesh.addPrimitive("vertices", numInstances=floor_positions.shape[0])
floor_position = floor_vertices.addAttribute("position", rows=3, cols=1)
floor_position.updateValue(floor_positions.reshape(-1))
floor_last_position = floor_vertices.addConstant("last_position", rows=3, cols=1)
floor_last_position.updateValue(floor_positions.reshape(-1))

floor_triangle_primitive = floor_mesh.addPrimitive("triangles", numInstances=floor_triangles.shape[0])
floor_triangles_to_vertices = floor_triangle_primitive.addConnectivity("triangles_to_vertices", floor_vertices, floor_triangles, 3)
floor_triangle_primitive.addAttribute("positions", through=floor_triangles_to_vertices, source=floor_position)


# Register the bunny and floor collision topology.

collision_mesh = simulation.addMesh("collision_mesh")
collision_vertices = collision_mesh.addPrimitiveUnion("vertices", [vertices_soft, floor_vertices])
collision_position = collision_vertices.addAttribute("position")

collision_mesh.addPrimitive("pp", numInstances=0, isDynamic=True)
collision_mesh.addPrimitive("pe", numInstances=0, isDynamic=True)
collision_mesh.addPrimitive("pt", numInstances=0, isDynamic=True)
collision_mesh.addPrimitive("ee", numInstances=0, isDynamic=True)

pp_to_vertices = collision_mesh.pp.addConnectivity("pp_to_vertices", collision_vertices, [], 2)
pe_to_vertices = collision_mesh.pe.addConnectivity("pe_to_vertices", collision_vertices, [], 3)
pt_to_vertices = collision_mesh.pt.addConnectivity("pt_to_vertices", collision_vertices, [], 4)
ee_to_vertices = collision_mesh.ee.addConnectivity("ee_to_vertices", collision_vertices, [], 4)

pp_positions = collision_mesh.pp.addAttribute("positions", through=pp_to_vertices, source=collision_position)
pe_positions = collision_mesh.pe.addAttribute("positions", through=pe_to_vertices, source=collision_position)
pt_positions = collision_mesh.pt.addAttribute("positions", through=pt_to_vertices, source=collision_position)
ee_positions = collision_mesh.ee.addAttribute("positions", through=ee_to_vertices, source=collision_position)



# Define inertia, elasticity, contact, friction, and loss energies.

elastic_energy = tets_soft.addAttribute("elastic_energy", computed_attribute=stable_neo_hookean(tet_rest_positions, tet_positions, mu, lam, dt))
inertia_energy = vertices_soft.addAttribute("inertia_energy", computed_attribute=inertia(last_position, velocity, dt, position, mass))
pp_energy = collision_mesh.pp.addAttribute("point_point_energy", computed_attribute=point_point(pp_positions, dhat, kappa))
pe_energy = collision_mesh.pe.addAttribute("point_edge_energy", computed_attribute=point_edge(pe_positions, dhat, kappa))
pt_energy = collision_mesh.pt.addAttribute("point_triangle_energy", computed_attribute=point_triangle(pt_positions, dhat, kappa))
ee_energy = collision_mesh.ee.addAttribute("edge_edge_energy", computed_attribute=edge_edge(ee_positions, dhat, kappa))
loss_offset = position - attribute.to_array(FLOOR_CENTER.tolist(), rows=3, cols=1)
terminal_loss = vertices_soft.addAttribute("terminal_position_loss", computed_attribute=loss_offset.dot(loss_offset) * (1.0 / num_bunny_vertices))
terminal_loss_gradient = differentiator().diff1([terminal_loss], [position])

simulation.addEnergy(elastic_energy, projection_method=1)
simulation.addEnergy(inertia_energy, projection_method=-1)
simulation.addEnergy(pp_energy, dynamic_instances=True, projection_method=2)
simulation.addEnergy(pe_energy, dynamic_instances=True, projection_method=2)
simulation.addEnergy(pt_energy, dynamic_instances=True, projection_method=2)
simulation.addEnergy(ee_energy, dynamic_instances=True, projection_method=2)
simulation.addMinimizeTarget([position])

# Configure continuous collision detection for each Newton step.
triangle_indices = np.vstack([bunny_surface_triangles, floor_triangles + floor_vertex_offset,]).astype(np.uint32)
edge_indices = np.vstack([bunny_edges, floor_edges + floor_vertex_offset,]).astype(np.uint32)
surface_indices = np.hstack([bunny_surface_indices, floor_surface_indices + floor_vertex_offset,]).astype(np.uint32)

initial_collision_positions = collision_position.compute().value.get()
ccd = CCD(surface_indices.size, num_bunny_vertices + floor_positions.shape[0], max_ccd_pairs=10000000, max_cd_pairs=30000000)
ccd_positions = gpuarray.to_gpu(initial_collision_positions)
ccd.init_faces(ccd_positions, gpuarray.to_gpu(triangle_indices.reshape(-1)), gpuarray.to_gpu(surface_indices.reshape(-1)), triangle_indices.shape[0])
ccd.init_edges(ccd_positions, ccd_positions, gpuarray.to_gpu(edge_indices.reshape(-1)), edge_indices.shape[0])


# Snapshot and restore dynamic contact state needed by the reverse pass.
def save_collision_pairs():
  pp_count = collision_mesh.pp.numInstances
  pe_count = collision_mesh.pe.numInstances
  pt_count = collision_mesh.pt.numInstances
  ee_count = collision_mesh.ee.numInstances
  return (
    ccd.pp[:2 * pp_count].get().copy(),
    ccd.pe[:3 * pe_count].get().copy(),
    ccd.pt[:4 * pt_count].get().copy(),
    ccd.ee[:4 * ee_count].get().copy(),
  )


def restore_collision_pairs(saved_pairs):
  pp, pe, pt, ee = saved_pairs
  pp_count = pp.size // 2
  pe_count = pe.size // 3
  pt_count = pt.size // 4
  ee_count = ee.size // 4

  collision_mesh.pp.updateNumInstances(pp_count)
  collision_mesh.pe.updateNumInstances(pe_count)
  collision_mesh.pt.updateNumInstances(pt_count)
  collision_mesh.ee.updateNumInstances(ee_count)

  if pp_count > 0:
    pp_to_vertices.updateConnectivity(pp.reshape((-1, 2)))
  if pe_count > 0:
    pe_to_vertices.updateConnectivity(pe.reshape((-1, 3)))
  if pt_count > 0:
    pt_to_vertices.updateConnectivity(pt.reshape((-1, 4)))
  if ee_count > 0:
    ee_to_vertices.updateConnectivity(ee.reshape((-1, 4)))

def update_collision_pairs():
  ccd.cd(collision_position.compute().value, DHAT_VALUE)
  pp_count, pe_count, pt_count, ee_count = (
    int(value) for value in ccd.separated_counts
  )

  collision_mesh.pp.updateNumInstances(pp_count)
  collision_mesh.pe.updateNumInstances(pe_count)
  collision_mesh.pt.updateNumInstances(pt_count)
  collision_mesh.ee.updateNumInstances(ee_count)

  if pp_count > 0:
    pp_to_vertices.updateConnectivity(ccd.pp[:2 * pp_count])
  if pe_count > 0:
    pe_to_vertices.updateConnectivity(ccd.pe[:3 * pe_count])
  if pt_count > 0:
    pt_to_vertices.updateConnectivity(ccd.pt[:4 * pt_count])
  if ee_count > 0:
    ee_to_vertices.updateConnectivity(ccd.ee[:4 * ee_count])

  return pp_count, pe_count, pt_count, ee_count

# Prebuild the projected Hessian used by the approximate adjoint solve.

def add_matrices(matrices):
  result = matrices[0]
  for current in matrices[1:]:
    result = result + current
  return result


# This is an approximate adjoint: use exactly the same per-energy Hessian
# projections as the forward Newton solve.  The resulting SPD matrix can be
# solved entirely by the existing GPU PCG solver without CPU assembly.
adjoint_hessian_terms = [
  differentiator().diff2([elastic_energy], [position], [position], projection_method=1),
  differentiator().diff2([inertia_energy], [position], [position], projection_method=-1),
  differentiator().diff2([pp_energy], [position], [position], projection_method=2, dynamic_instances=True),
  differentiator().diff2([pe_energy], [position], [position], projection_method=2, dynamic_instances=True),
  differentiator().diff2([pt_energy], [position], [position], projection_method=2, dynamic_instances=True),
  differentiator().diff2([ee_energy], [position], [position], projection_method=2, dynamic_instances=True),
]
adjoint_hessian = add_matrices(adjoint_hessian_terms)

# Build residual Jacobians used to propagate adjoints and design gradients.

previous_position_jacobian = differentiator().diff2([inertia_energy], [position], [last_position])
previous_previous_position_jacobian = differentiator().diff2([inertia_energy], [position], [last_last_position])
initial_translation_jacobian = differentiator().diff1(initial_position_mapping, [initial_translation_parameter])
adjoint_linear_solver = solver()
adjoint_initial_guess = gpuarray.zeros(3 * num_bunny_vertices, np.float64)


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

# Configure rendering and surface-only output.
bunny_cells = np.hstack([np.full((bunny_surface_triangles.shape[0], 1), 3, dtype=np.uint32), bunny_surface_triangles])
floor_cells = np.hstack([np.full((floor_triangles.shape[0], 1), 3, dtype=np.uint32), floor_triangles])

bunny_poly = pv.PolyData(bunny_positions, bunny_cells)
floor_poly = pv.PolyData(floor_positions, floor_cells)

plotter = pv.Plotter(window_size=[1920, 1080], off_screen=args.no_gui)
plotter.add_mesh(bunny_poly, color="lightgreen", opacity=0.65, smooth_shading=True, show_edges=False)
plotter.add_mesh(floor_poly, color="lightgray", opacity=0.25, show_edges=True)
floor_center_marker = pv.Sphere(radius=0.05, center=FLOOR_CENTER, theta_resolution=32, phi_resolution=32)
plotter.add_mesh(floor_center_marker, color="red", ambient=1.0, pickable=False,)
plotter.add_text("Initial configuration", name="optimization_status", position="upper_left", font_size=16, color="black")
plotter.camera_position = [
  (3.0, 2.5, 6.0),
  (0.0, 0.75, 0.0),
  (0.0, 1.0, 0.0),
]

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

if not args.no_gui:
  plotter.show(interactive_update=True, auto_close=False)

default_output_name = "inverse_initial_position_projected_adam"
output_directory = args.output_directory or os.path.join(OUTPUT_ROOT, "outputs", default_output_name,)
output_directory = os.path.abspath(output_directory)
os.makedirs(output_directory, exist_ok=True)


def refresh_gui():
  current_positions = position.compute().value.get().reshape((-1, 3))
  bunny_poly.points = current_positions
  plotter.render()
  if not args.no_gui:
    plotter.update()
  return current_positions


def output_paths(adjoint_round):
  round_name = (
    "baseline"
    if adjoint_round == 0
    else f"adjoint_{adjoint_round:02d}"
  )
  round_directory = os.path.join(output_directory, round_name)
  return (
    round_name,
    os.path.join(round_directory, "frames"),
    os.path.join(round_directory, "bunny_obj"),
    os.path.join(output_directory, f"{round_name}.mp4"),
  )


def encode_saved_frames(adjoint_round, loss, baseline_loss=None):
  round_name, frame_directory, _, video_path = output_paths(adjoint_round)
  ffmpeg = shutil.which("ffmpeg")
  if ffmpeg is None:
    raise RuntimeError("--save-frames requires ffmpeg to create the final MP4.")

  if adjoint_round == 0:
    loss_text = f"unoptimized baseline   loss {loss:.8f}"
  else:
    loss_text = (
      f"adjoint {adjoint_round:02d}/{args.adjoint_steps:02d}   "
      f"loss {loss:.8f}   baseline {baseline_loss:.8f}   "
      f"relative reduction {(baseline_loss - loss) / baseline_loss:.6f}"
    )

  ffmpeg_command = [
      ffmpeg, "-y", "-loglevel", "error", "-framerate", str(args.video_fps),
      "-start_number", "0", "-i", os.path.join(frame_directory, "frame_%04d.png"),
      "-frames:v", str(args.num_frames), "-vf", (
        "drawtext="
        f"text='{loss_text}':"
        "fontcolor=black:fontsize=36:"
        "box=1:boxcolor=white@0.78:boxborderw=12:"
        "x=40:y=h-th-40"
      ),
      "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", video_path,
  ]
  subprocess.run(ffmpeg_command, check=True)
  print(f"Saved {round_name} video: {video_path}")
  return video_path


# Run one complete forward trajectory and record every converged state.
def run_forward(design_parameters, adjoint_round=None, baseline_loss=None):
  forward_index = 0 if adjoint_round is None else adjoint_round
  max_newton_iterations = (
    REFRESH_MAX_NEWTON_ITERATIONS
    if forward_index % REFRESH_NEWTON_EVERY == 0
    else MAX_NEWTON_ITERATIONS
  )
  initial_translation = design_parameters
  initial_translation_parameter.updateValue(initial_translation)
  initial_position_mapping.compute()
  initial_position = initial_position_mapping.value.get().reshape((-1, 3))
  position.updateValue(initial_position.reshape(-1))
  last_position.updateValue(initial_position.reshape(-1))
  last_last_position.updateValue(initial_position.reshape(-1))
  update_collision_pairs()
  refresh_gui()

  saved_positions = [initial_position.copy()]
  saved_collision_pairs = []
  saved_video = None

  save_this_forward = adjoint_round is not None
  if save_this_forward:
    _, frame_directory, obj_directory, _ = output_paths(adjoint_round)
    design_status = (
      f"initial x/z translation = "
      f"({initial_translation[0]:.5f}, "
      f"{initial_translation[2]:.5f})"
    )
    status_title = (
      "Unoptimized baseline"
      if adjoint_round == 0
      else f"After adjoint {adjoint_round:02d}/{args.adjoint_steps:02d}"
    )
    plotter.add_text(f"{status_title}\n{design_status}", name="optimization_status", position="upper_left", font_size=16, color="black")
    if args.save_frames:
      os.makedirs(frame_directory, exist_ok=True)
    if args.save_obj:
      os.makedirs(obj_directory, exist_ok=True)

  collision_position_copy = collision_position.compute().value.copy()
  bunny_position_copy = position.compute().value.copy()
  gpu_start_used, gpu_total = gpu_memory_used()
  gpu_peak_used = gpu_start_used
  forward_start = time.time()

  for frame in range(args.num_frames):
    last_last_position.updateValue(last_position.value, deepCopy=True)
    last_position.updateValue(position.value, deepCopy=True)

    converged = False
    for newton_iteration in range(max_newton_iterations):
      collision_counts_before = update_collision_pairs()
      energy_before = simulation.computeTotalEnergy()

      solve_start = time.time()
      result = simulation.minimizeEnergy(tolerance=SOLVER_TOLERANCE, maxIterations=MAX_CG_ITERATIONS)
      gpu_current_used, _ = gpu_memory_used()
      gpu_peak_used = max(gpu_peak_used, gpu_current_used)
      solve_duration = time.time() - solve_start
      displacement = result[0]

      collision_position_copy.set(collision_position.compute().value.copy())
      bunny_position_copy.set(position.compute().value)

      position.updateValue(bunny_position_copy - displacement, deepCopy=True)
      proposed_collision_position = collision_position.compute().value
      direction = collision_position_copy - proposed_collision_position
      max_movement = (
        float(gpuarray.max(abs(direction)).get()) / DT_VALUE
      )

      # compute ccd to determine the largest step we can take
      ccd.ccd(collision_position_copy, DHAT_VALUE, direction, 0.5)
      largest_step = float(ccd.compute_largest_step_size(0.5, collision_position_copy, direction))

      step_taken = largest_step
      energy_after = energy_before
      collision_counts_after = collision_counts_before

      for _ in range(MAX_LINE_SEARCH_STEPS):
        position.updateValue(bunny_position_copy - displacement * step_taken, deepCopy=True)
        collision_counts_after = update_collision_pairs()
        energy_after = simulation.computeTotalEnergy()

        if energy_after <= energy_before:
          break
        step_taken *= 0.5

      prefix = "initial" if adjoint_round in (None, 0) else f"adjoint={adjoint_round:02d}"
      iteration_message = (
        f"{prefix} frame={frame:03d} newton={newton_iteration:03d} "
        f"solver={solve_duration:.4f}s step={step_taken:.6f} "
        f"max_movement={max_movement:.6f} "
        f"energy={energy_before:.8e}->{energy_after:.8e} "
        f"pairs={collision_counts_after}"
      )
      print(iteration_message)

      if not args.no_gui:
        refresh_gui()

      if max_movement < MOTION_TOLERANCE:
        converged = True
        break

    if not converged:
      print(f"Warning: frame {frame} reached the " f"{max_newton_iterations}-iteration Newton cap.")

    update_collision_pairs()
    current_positions = refresh_gui()

    # Only converged state data are checkpointed.  Numeric Hessian and
    # Jacobian values are deliberately not stored; the reverse loop restores
    # these inputs and recomputes all matrices on demand.
    saved_positions.append(current_positions.copy())
    saved_collision_pairs.append(save_collision_pairs())

    if save_this_forward and args.save_frames:
      plotter.screenshot(os.path.join(frame_directory, f"frame_{frame:04d}.png",))
    if save_this_forward and args.save_obj:
      bunny_poly.save(os.path.join(obj_directory, f"bunny_{frame:04d}.obj",))

  # here we compute the loss, and the gradient of loss wrt the final position
  loss = float(terminal_loss.compute().value.get().sum())
  terminal_loss_gradient.compute()
  final_loss_gradient = vector(terminal_loss_gradient.size)
  final_loss_gradient.updateValue(terminal_loss_gradient)
  gpu_end_used, _ = gpu_memory_used()
  gpu_peak_used = max(gpu_peak_used, gpu_end_used)
  memory = gpu_memory_summary(gpu_start_used, gpu_peak_used, gpu_end_used, gpu_total)
  elapsed = time.time() - forward_start
  print(f"Finished forward " f"{'initial' if adjoint_round in (None, 0) else adjoint_round} in " f"{elapsed:.2f}s: loss={loss:.8e}, " f"newton_cap={max_newton_iterations}, gpu_memory={memory}.")

  if save_this_forward and args.save_frames:
    saved_video = encode_saved_frames(adjoint_round, loss, baseline_loss=baseline_loss,)

  return {
    "positions": saved_positions,
    "collision_pairs": saved_collision_pairs,
    "loss": loss,
    "final_loss_gradient": final_loss_gradient,
    "elapsed_seconds": elapsed,
    "max_newton_iterations": max_newton_iterations,
    "gpu_memory": memory,
    "video": saved_video,
  }


# Replay the saved trajectory backward to accumulate adjoints and design gradients.
def run_backward(trajectory, adjoint_round):
  # The position-only state is second order: step k depends on q[k] and
  # q[k - 1].  Keep an explicitly indexed accumulator so that the two
  # residual pullbacks are placed into different time indices.  Index i + 1
  # stores the adjoint of q[i], including index 0 for the synthetic q[-1]
  # used to impose the initial zero velocity.
  position_adjoints = [
    vector(trajectory["final_loss_gradient"].size)
    for _ in range(args.num_frames + 2)
  ]
  position_adjoints[args.num_frames + 1].updateValue(trajectory["final_loss_gradient"])
  gpu_start_used, gpu_total = gpu_memory_used()
  gpu_peak_used = gpu_start_used
  backward_start = time.time()
  timing = {
    "state_restore_seconds": 0.0,
    "projected_hessian_seconds": 0.0,
    "gpu_cg_seconds": 0.0,
    "jacobian_and_transpose_spmv_seconds": 0.0,
  }

  for frame in range(args.num_frames - 1, -1, -1):
    restore_start = time.time()
    last_position.updateValue(trajectory["positions"][frame].reshape(-1))
    last_last_position.updateValue(trajectory["positions"][max(0, frame - 1)].reshape(-1))
    position.updateValue(trajectory["positions"][frame + 1].reshape(-1))
    restore_collision_pairs(trajectory["collision_pairs"][frame])
    timing["state_restore_seconds"] += time.time() - restore_start

    # q[k + 1] is stored at adjoint index k + 2.
    # This solves mu[k+1] = A[k+1]^{-T} lambda[k+1].
    auxiliary, hessian_seconds, solve_seconds = solve_adjoint_system(position_adjoints[frame + 2])
    timing["projected_hessian_seconds"] += hessian_seconds
    timing["gpu_cg_seconds"] += solve_seconds

    jacobian_start = time.time()
    previous_position_jacobian.compute()
    previous_position_contribution = previous_position_jacobian.spmv(auxiliary, transpose=True)
    previous_previous_position_jacobian.compute()
    previous_previous_position_contribution = previous_previous_position_jacobian.spmv(auxiliary, transpose=True)
    timing["jacobian_and_transpose_spmv_seconds"] += (
      time.time() - jacobian_start
    )

    # B[k+1]^T mu contributes to q[k], while D[k+1]^T mu contributes
    # to q[k - 1].  Both carry the minus sign from implicit differentiation.
    position_adjoints[frame + 1] = (
      position_adjoints[frame + 1]
      - previous_position_contribution
    )
    position_adjoints[frame] = (
      position_adjoints[frame]
      - previous_previous_position_contribution
    )
    gpu_current_used, _ = gpu_memory_used()
    gpu_peak_used = max(gpu_peak_used, gpu_current_used)

    if (
      frame == args.num_frames - 1
      or frame == 0
      or frame % max(1, args.num_frames // 10) == 0
    ):
      print(f"adjoint={adjoint_round:02d} reverse_frame={frame:03d} " f"|lambda_x|_inf={float(gpuarray.max(abs(position_adjoints[frame + 1].value)).get()):.6e} " "solver=gpu_cg")

  # q[-1] and q[0] are both initialized by the same map Gamma(theta), so
  # their independently accumulated adjoints both contribute to theta.
  initial_state_adjoint = position_adjoints[0] + position_adjoints[1]
  initial_translation_jacobian.compute()
  gradient = initial_translation_jacobian.spmv(initial_state_adjoint, transpose=True).value.get()
  gradient_summary = f"translation_gradient={gradient}"
  elapsed = time.time() - backward_start
  gpu_end_used, _ = gpu_memory_used()
  gpu_peak_used = max(gpu_peak_used, gpu_end_used)
  memory = gpu_memory_summary(gpu_start_used, gpu_peak_used, gpu_end_used, gpu_total,)
  timing["other_seconds"] = elapsed - sum(timing.values())
  print(f"Finished adjoint {adjoint_round:02d} in {elapsed:.2f}s: " f"{gradient_summary}, timing={timing}, gpu_memory={memory}.")
  return gradient, elapsed, timing, memory


update_collision_pairs()
refresh_gui()

# Initialize or restore the design, then alternate one backward and one forward solve.
optimization_start = time.time()
design_parameters = np.zeros(3, dtype=np.float64)
baseline_trajectory = run_forward(design_parameters, adjoint_round=0)
trajectory = baseline_trajectory
initial_loss = trajectory["loss"]
best_loss = initial_loss
best_adjoint_round = 0
best_design_parameters = design_parameters.copy()
optimization_history = []
maximum_step_size = args.initial_position_step_size
adam_first_moment = np.zeros_like(design_parameters)
adam_second_moment = np.zeros_like(design_parameters)

for adjoint_round in range(1, args.adjoint_steps + 1):
  gradient, backward_seconds, backward_timing, backward_memory = run_backward(trajectory, adjoint_round)
  gradient_values = gradient

  constrained_gradient = gradient_values.copy()
  constrained_gradient[1] = 0.0
  gradient_norm = float(np.linalg.norm(constrained_gradient))
  if not np.all(np.isfinite(constrained_gradient)) or gradient_norm == 0.0:
    print("Stopping because the design gradient is zero or non-finite.")
    break

  previous_loss = trajectory["loss"]
  previous_design = design_parameters.copy()
  adam_first_moment = ADAM_BETA1 * adam_first_moment + (1.0 - ADAM_BETA1) * constrained_gradient
  adam_second_moment = ADAM_BETA2 * adam_second_moment + (1.0 - ADAM_BETA2) * constrained_gradient * constrained_gradient
  corrected_first_moment = adam_first_moment / (1.0 - ADAM_BETA1 ** adjoint_round)
  corrected_second_moment = adam_second_moment / (1.0 - ADAM_BETA2 ** adjoint_round)
  raw_update = -args.adam_learning_rate * corrected_first_moment / (np.sqrt(corrected_second_moment) + ADAM_EPSILON)
  raw_update[1] = 0.0
  raw_update_norm = float(np.linalg.norm(raw_update))
  applied_step = min(maximum_step_size, raw_update_norm) * raw_update / raw_update_norm
  design_parameters = design_parameters + applied_step

  trajectory = run_forward(design_parameters, adjoint_round=adjoint_round, baseline_loss=initial_loss,)
  loss_decreased = trajectory["loss"] < previous_loss

  record = {
    "adjoint_round": adjoint_round,
    "loss_before": previous_loss,
    "loss_after": trajectory["loss"],
    "loss_decreased": loss_decreased,
    "forward_seconds": trajectory["elapsed_seconds"],
    "forward_max_newton_iterations": trajectory["max_newton_iterations"],
    "forward_gpu_memory": trajectory["gpu_memory"],
    "backward_seconds": backward_seconds,
    "backward_timing": backward_timing,
    "backward_gpu_memory": backward_memory,
    "adjoint_solver": "gpu_cg",
    "adjoint_solver_count": args.num_frames,
    "video": trajectory["video"],
    "gradient": gradient_values.tolist(),
    "raw_adam_update": raw_update.tolist(),
    "applied_step": applied_step.tolist(),
    "design_parameters_before": previous_design.tolist(),
    "design_parameters_after": design_parameters.tolist(),
    "initial_translation_before": previous_design.tolist(),
    "initial_translation_after": design_parameters.tolist(),
  }
  design_summary = f"translation={previous_design}->{design_parameters}"
  optimization_history.append(record)
  if trajectory["loss"] < best_loss:
    best_loss = trajectory["loss"]
    best_adjoint_round = adjoint_round
    best_design_parameters = design_parameters.copy()
  print(f"optimization={adjoint_round:02d}/{args.adjoint_steps:02d} " f"loss={previous_loss:.8e}->{trajectory['loss']:.8e} " f"{design_summary} " f"step_norm={np.linalg.norm(applied_step):.6e}")

# Save final metrics and the best/current parameter arrays for later inspection.
results = {
  "num_frames": args.num_frames,
  "dt": DT_VALUE,
  "dhat": DHAT_VALUE,
  "newton_schedule": {
    "default_max_iterations": MAX_NEWTON_ITERATIONS,
    "refresh_max_iterations": REFRESH_MAX_NEWTON_ITERATIONS,
    "refresh_every": REFRESH_NEWTON_EVERY,
  },
  "requested_adjoint_steps": args.adjoint_steps,
  "completed_adjoint_steps": len(optimization_history),
  "optimization_target": "initial-position",
  "adjoint_type": "approximate_projected_hessian",
  "adjoint_solver": "gpu_cg",
  "optimizer": "adam",
  "initial_position_step_size": args.initial_position_step_size,
  "adam_learning_rate": args.adam_learning_rate,
  "adam_beta1": ADAM_BETA1,
  "adam_beta2": ADAM_BETA2,
  "adam_epsilon": ADAM_EPSILON,
  "loss_definition": (
    "mean squared distance from every final bunny vertex to floor center"
  ),
  "floor_center": FLOOR_CENTER.tolist(),
  "initial_loss": initial_loss,
  "baseline_max_newton_iterations": baseline_trajectory["max_newton_iterations"],
  "baseline_gpu_memory": baseline_trajectory["gpu_memory"],
  "baseline_video": baseline_trajectory["video"],
  "best_loss": best_loss,
  "best_adjoint_round": best_adjoint_round,
  "final_loss": trajectory["loss"],
  "best_design_parameters": best_design_parameters.tolist(),
  "final_design_parameters": design_parameters.tolist(),
  "best_initial_translation": best_design_parameters.tolist(),
  "final_initial_translation": design_parameters.tolist(),
  "history": optimization_history,
  "total_seconds": time.time() - optimization_start,
}
results_path = os.path.join(output_directory, "results.json")
with open(results_path, "w", encoding="utf-8") as file:
  json.dump(results, file, indent=2)

print(f"Inverse simulation finished in {results['total_seconds']:.2f}s: " f"loss={initial_loss:.8e}->{trajectory['loss']:.8e}, " "target=initial-position.")
print(f"Saved optimization results: {results_path}")
if not args.no_gui:
  plotter.close()
