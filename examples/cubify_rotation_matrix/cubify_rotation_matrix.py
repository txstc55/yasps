import argparse
import os
import shutil
import subprocess
import time

import pyvista as pv

from yasps import minimizer, scene

from helpers import build_geometry, build_triangle_cells, edge_arap_energy, load_triangle_obj, normalize_bunny, place_on_ground, save_obj_without_normals, vertex_cubic_energy, vertex_determinant_energy, vertex_inertia_energy, vertex_orthogonality_energy, vertex_regularization_energy


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MESH = os.path.abspath(os.path.join(SCRIPT_DIR, "../data/bunny_small.obj"))
DEFAULT_OUTPUT_DIRECTORY = os.path.join(SCRIPT_DIR, "outputs", "bunny_matrix_rotation_lambda_10")

CUBIC_WEIGHT = 10.0
ARAP_WEIGHT = 1.0
ORTHOGONALITY_WEIGHT = 1e4
DETERMINANT_WEIGHT = 1e4
POSITION_REGULARIZATION_WEIGHT = 1e-12
MASS_SCALE = 0.001
TIME_STEP = 0.01
GRAVITY = 0.0
NUM_FRAMES = 10
MAX_ROTATION_NEWTON_ITERATIONS = 100
MAX_POSITION_NEWTON_ITERATIONS = 300
NEWTON_CONVERGENCE_TOLERANCE = 1e-3
LINE_SEARCH_STEPS = 8
SOLVER_TOLERANCE = 1e-6
MAX_CG_ITERATIONS = 20000
VIDEO_FPS = 100


parser = argparse.ArgumentParser(description="Forward cubic stylization with relaxed 3x3 rotation matrices.")
parser.add_argument("--mesh", default=DEFAULT_MESH)
parser.add_argument("--num-frames", type=int, default=NUM_FRAMES)
parser.add_argument("--max-rotation-newton-iterations", type=int, default=MAX_ROTATION_NEWTON_ITERATIONS)
parser.add_argument("--max-position-newton-iterations", type=int, default=MAX_POSITION_NEWTON_ITERATIONS)
parser.add_argument("--cubic-weight", type=float, default=CUBIC_WEIGHT)
parser.add_argument("--orthogonality-weight", type=float, default=ORTHOGONALITY_WEIGHT)
parser.add_argument("--determinant-weight", type=float, default=DETERMINANT_WEIGHT)
parser.add_argument("--time-step", type=float, default=TIME_STEP)
parser.add_argument("--save-frames", action="store_true")
parser.add_argument("--save-obj", action="store_true")
parser.add_argument("--no-gui", action="store_true")
parser.add_argument("--video-fps", type=int, default=VIDEO_FPS)
parser.add_argument("--output-directory", default=DEFAULT_OUTPUT_DIRECTORY)
args = parser.parse_args()

if args.num_frames < 0:
  raise ValueError("--num-frames must be non-negative.")
if args.max_rotation_newton_iterations <= 0 or args.max_position_newton_iterations <= 0:
  raise ValueError("Newton iteration limits must be positive.")
if min(args.cubic_weight, args.orthogonality_weight, args.determinant_weight) < 0.0:
  raise ValueError("Energy weights must be non-negative.")
if args.time_step <= 0.0 or args.video_fps <= 0:
  raise ValueError("The time step and video frame rate must be positive.")

newton_convergence_tolerance = NEWTON_CONVERGENCE_TOLERANCE


##################################################################
## Load and preprocess the bunny surface mesh
##################################################################
vertices, triangles = load_triangle_obj(os.path.abspath(args.mesh))
rest_vertices = normalize_bunny(vertices)
vertex_normals, vertex_areas, edges, cotangent_weights, nonpositive_weights, degenerate_faces = build_geometry(rest_vertices, triangles)
num_vertices, num_triangles, num_edges = rest_vertices.shape[0], triangles.shape[0], edges.shape[0]
print(f"Loaded {num_vertices} vertices, {num_triangles} triangles, and {num_edges} edges; clamped {nonpositive_weights} non-positive cotangent weights and skipped {degenerate_faces} degenerate faces.")


##################################################################
## Construct the YASPS scene and relaxed rotation-matrix field
##################################################################
model = scene("bunny_cubic_stylization_matrix_rotation")
cubic_weight = model.addConstant("cubic_weight", rows=1, cols=1)
arap_weight = model.addConstant("arap_weight", rows=1, cols=1)
orthogonality_weight = model.addConstant("orthogonality_weight", rows=1, cols=1)
determinant_weight = model.addConstant("determinant_weight", rows=1, cols=1)
regularization_weight = model.addConstant("position_regularization_weight", rows=1, cols=1)
time_step = model.addConstant("time_step", rows=1, cols=1)
cubic_weight.updateValue([args.cubic_weight])
arap_weight.updateValue([ARAP_WEIGHT])
orthogonality_weight.updateValue([args.orthogonality_weight])
determinant_weight.updateValue([args.determinant_weight])
regularization_weight.updateValue([POSITION_REGULARIZATION_WEIGHT])
time_step.updateValue([args.time_step])

bunny = model.addMesh("bunny")
vertex_primitive = bunny.addPrimitive("vertices", numInstances=num_vertices)
position = vertex_primitive.addAttribute("position", rows=3, cols=1)
position.updateValue(rest_vertices)
position_constant = vertex_primitive.addConstant("position_constant", rows=3, cols=1)
position_constant.updateValue(rest_vertices)
rest_position = vertex_primitive.addConstant("rest_position", rows=3, cols=1)
rest_position.updateValue(rest_vertices)
normal = vertex_primitive.addConstant("normal", rows=3, cols=1)
normal.updateValue(vertex_normals)
area = vertex_primitive.addConstant("barycentric_area", rows=1, cols=1)
area.updateValue(vertex_areas)
mass = vertex_primitive.addConstant("mass", rows=1, cols=1)
vertex_masses = MASS_SCALE * vertex_areas
mass.updateValue(vertex_masses)
last_position = vertex_primitive.addConstant("last_position", rows=3, cols=1)
last_position.updateValue(rest_vertices)
velocity = vertex_primitive.addConstant("velocity", rows=3, cols=1)
velocity.updateValue([0.0] * (3 * num_vertices))
rotation = vertex_primitive.addAttribute("rotation", rows=3, cols=3)
rotation.updateValue([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] * num_vertices)
rotation_constant = vertex_primitive.addConstant("rotation_constant", rows=3, cols=3)
rotation_constant.updateValue(rotation.value, deepCopy=True)

edge_primitive = bunny.addPrimitive("edges", numInstances=num_edges)
edge_to_vertex = edge_primitive.addConnectivity("edge_to_vertex", vertex_primitive, edges.flatten(), 2)
edge_weight = edge_primitive.addConstant("cotangent_weight", rows=1, cols=1)
edge_weight.updateValue(cotangent_weights)
edge_positions = edge_primitive.addAttribute("positions", through=edge_to_vertex, source=position)
edge_positions_constant = edge_primitive.addAttribute("positions_constant", through=edge_to_vertex, source=position_constant)
edge_rest_positions = edge_primitive.addAttribute("rest_positions", through=edge_to_vertex, source=rest_position)
edge_rotations = edge_primitive.addAttribute("rotations", through=edge_to_vertex, source=rotation)
edge_rotations_constant = edge_primitive.addAttribute("rotations_constant", through=edge_to_vertex, source=rotation_constant)


##################################################################
## Construct the local relaxed-rotation minimization
##################################################################
cubic_energy = vertex_primitive.addAttribute("cubic_energy", computed_attribute=vertex_cubic_energy(area, rotation, normal, cubic_weight))
orthogonality_energy = vertex_primitive.addAttribute("orthogonality_energy", computed_attribute=vertex_orthogonality_energy(area, rotation, orthogonality_weight))
determinant_energy = vertex_primitive.addAttribute("determinant_energy", computed_attribute=vertex_determinant_energy(area, rotation, determinant_weight))
local_arap_energy = edge_primitive.addAttribute("local_arap_energy", computed_attribute=edge_arap_energy(edge_weight, edge_positions_constant, edge_rotations, edge_rest_positions, arap_weight))
local_minimizer = minimizer()
local_minimizer.addEnergy(cubic_energy, projection_method=2)
local_minimizer.addEnergy(local_arap_energy, projection_method=2)
local_minimizer.addEnergy(orthogonality_energy, projection_method=1)
local_minimizer.addEnergy(determinant_energy, projection_method=1)
local_minimizer.addWrt([rotation])
local_minimizer.generateHessianAndGradient()


##################################################################
## Construct the global implicit-Euler position minimization
##################################################################
global_arap_energy = edge_primitive.addAttribute("dynamic_global_arap_energy", computed_attribute=edge_arap_energy(edge_weight, edge_positions, edge_rotations_constant, edge_rest_positions, arap_weight) * (args.time_step * args.time_step))
regularization_energy = vertex_primitive.addAttribute("dynamic_regularization_energy", computed_attribute=vertex_regularization_energy(position, rest_position, area, regularization_weight) * (args.time_step * args.time_step))
inertia_energy = vertex_primitive.addAttribute("inertia_energy", computed_attribute=vertex_inertia_energy(last_position, velocity, time_step, position, mass, GRAVITY))
global_minimizer = minimizer()
global_minimizer.addEnergy(global_arap_energy, projection_method=1)
global_minimizer.addEnergy(regularization_energy, projection_method=1)
global_minimizer.addEnergy(inertia_energy, projection_method=0)
global_minimizer.addWrt([position])
global_minimizer.generateHessianAndGradient()


##################################################################
## Construct the renderer and output directories
##################################################################
output_directory = os.path.abspath(args.output_directory)
frame_directory = os.path.join(output_directory, "frames")
os.makedirs(output_directory, exist_ok=True)
if args.save_frames:
  os.makedirs(frame_directory, exist_ok=True)
if args.save_obj:
  save_obj_without_normals(os.path.join(output_directory, "bunny_normalized.obj"), rest_vertices, triangles)

cells = build_triangle_cells(triangles)
bunny_poly = pv.PolyData(rest_vertices, cells)
plotter = pv.Plotter(window_size=[1920, 1080], off_screen=args.no_gui)
plotter.set_background("#f3f1eb")
plotter.add_mesh(bunny_poly, color="#d78b58", smooth_shading=False, show_edges=True, edge_color="#5b3827", line_width=0.35)
plotter.camera_position = [(3.4, 2.3, 4.2), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
plotter.camera.zoom(1.25)
if not args.no_gui:
  plotter.show(interactive_update=True, auto_close=False)

start_time = time.time()

# Render the undeformed bunny before starting the implicit-Euler loop.
local_energy_value = local_minimizer.computeTotalEnergy()
global_energy_value = global_minimizer.computeTotalEnergy()
bunny_poly.points = position.value.get().reshape((-1, 3))
status = f"Matrix rotation cubic stylization  {0:03d}/{args.num_frames:03d}\nlocal {local_energy_value:.6e}   global {global_energy_value:.6e}"
plotter.add_text(status, name="status", position="upper_left", font_size=15, color="black")
plotter.render()
if not args.no_gui:
  plotter.update()
if args.save_frames:
  plotter.screenshot(os.path.join(frame_directory, "frame_0000.png"))

completed_iterations = 0


##################################################################
## Alternate local rotation and global position solves per frame
##################################################################
for outer_iteration in range(1, args.num_frames + 1):
  last_position.updateValue(position.value, deepCopy=True)
  local_maximum_solution = 0.0
  global_maximum_solution = 0.0
  rotation_newton_iterations = 0
  position_newton_iterations = 0
  rotation_converged = False
  position_converged = False

  for position_newton_iteration in range(args.max_position_newton_iterations):
    position_newton_iterations = position_newton_iteration + 1
    position_constant.updateValue(position.value, deepCopy=True)
    rotation_converged = False
    for _ in range(args.max_rotation_newton_iterations):
      rotation_newton_iterations += 1

      # Compute one Newton direction for the relaxed per-vertex rotation matrices.
      local_energy_before = local_minimizer.computeTotalEnergy()
      local_hessian = local_minimizer.computeNumericValue()
      local_error_code = local_minimizer.linearSolver.computeSolution(local_hessian, local_hessian.gradient, None, tolerance=SOLVER_TOLERANCE, maxIterations=MAX_CG_ITERATIONS, zero_initial_guess=True)
      if local_error_code < 0:
        print(f"Warning: local matrix solver returned error code {local_error_code}.")
      local_direction = local_minimizer.linearSolver.solution.get()
      local_maximum_solution = float(abs(local_direction).max())
      original_rotation = rotation.value.get()
      local_step_size = 1.0
      accepted_local = False

      # Backtrack along the Newton direction until the local energy decreases.
      for _ in range(LINE_SEARCH_STEPS):
        rotation.updateValue(original_rotation - local_step_size * local_direction)
        local_energy_value = local_minimizer.computeTotalEnergy()
        if local_energy_value <= local_energy_before:
          accepted_local = True
          break
        local_step_size *= 0.5

      # At an L1 kink, keep the final tiny Newton update rather than changing directions.
      if not accepted_local:
        rotation.updateValue(original_rotation - local_step_size * local_direction)
        local_energy_value = local_minimizer.computeTotalEnergy()

      # Test convergence only after the line search has applied its update.
      if local_maximum_solution < newton_convergence_tolerance:
        rotation_converged = True
        break
    if not rotation_converged:
      print(f"Warning: frame {outer_iteration} position iteration {position_newton_iteration} reached the {args.max_rotation_newton_iterations}-iteration matrix Newton cap with |solution|_inf={local_maximum_solution:.8e}.")

    # Freeze the optimized matrices and compute one global position Newton step.
    rotation_constant.updateValue(rotation.value, deepCopy=True)
    global_energy_before = global_minimizer.computeTotalEnergy()
    global_hessian = global_minimizer.computeNumericValue()
    global_error_code = global_minimizer.linearSolver.computeSolution(global_hessian, global_hessian.gradient, None, tolerance=SOLVER_TOLERANCE, maxIterations=MAX_CG_ITERATIONS, zero_initial_guess=True)
    if global_error_code < 0:
      print(f"Warning: global position solver returned error code {global_error_code}.")
    global_direction = global_minimizer.linearSolver.solution.get()
    global_maximum_solution = float(abs(global_direction).max())
    original_position = position.value.get()
    global_step_size = 1.0
    accepted_global = False

    # Backtrack without explicitly changing the assembled sparse system.
    for _ in range(LINE_SEARCH_STEPS):
      position.updateValue(original_position - global_step_size * global_direction)
      global_energy_value = global_minimizer.computeTotalEnergy()
      if global_energy_value <= global_energy_before:
        accepted_global = True
        break
      global_step_size *= 0.5

    if not accepted_global:
      position.updateValue(original_position)
      raise RuntimeError(f"Quadratic global Newton line search failed at frame {outer_iteration}, iteration {position_newton_iteration}.")
    if global_maximum_solution < newton_convergence_tolerance:
      position_converged = True
      break

  if not position_converged:
    print(f"Warning: frame {outer_iteration} reached the {args.max_position_newton_iterations}-iteration position Newton cap with |solution|_inf={global_maximum_solution:.8e}.")

  new_velocity = (position.value - last_position.value) / args.time_step
  velocity.updateValue(new_velocity, deepCopy=True)
  current_vertices = position.value.get().reshape((-1, 3))
  completed_iterations = outer_iteration

  # Render the converged implicit-Euler frame and record it when requested.
  bunny_poly.points = current_vertices
  status = f"Matrix rotation cubic stylization  {outer_iteration:03d}/{args.num_frames:03d}\nlocal {local_energy_value:.6e}   global {global_energy_value:.6e}"
  plotter.add_text(status, name="status", position="upper_left", font_size=15, color="black")
  plotter.render()
  if not args.no_gui:
    plotter.update()
  if args.save_frames:
    plotter.screenshot(os.path.join(frame_directory, f"frame_{outer_iteration:04d}.png"))
  print(f"frame={outer_iteration:03d}/{args.num_frames:03d} local={local_energy_value:.8e} global={global_energy_value:.8e} rotation_iterations={rotation_newton_iterations} rotation_solution={local_maximum_solution:.3e} position_iterations={position_newton_iterations} position_solution={global_maximum_solution:.3e}")


##################################################################
## Export the final mesh and optional video
##################################################################
elapsed_seconds = time.time() - start_time
final_vertices = place_on_ground(position.value.get().reshape((-1, 3)))
final_obj_path = os.path.join(output_directory, "bunny_cubic_stylized.obj")
save_obj_without_normals(final_obj_path, final_vertices, triangles)

video_path = None
if args.save_frames:
  ffmpeg = shutil.which("ffmpeg")
  if ffmpeg is None:
    raise RuntimeError("--save-frames requires ffmpeg to encode the MP4.")
  video_path = os.path.join(output_directory, "bunny_matrix_rotation_cubic_stylization.mp4")
  subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", str(args.video_fps), "-start_number", "0", "-i", os.path.join(frame_directory, "frame_%04d.png"), "-frames:v", str(completed_iterations + 1), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", video_path], check=True)

plotter.close()
local_minimizer.linearSolver.reset()
global_minimizer.linearSolver.reset()
print(f"Finished {completed_iterations} frames in {elapsed_seconds:.2f}s.")
print(f"Saved stylized bunny: {final_obj_path}")
if video_path is not None:
  print(f"Saved forward video: {video_path}")
