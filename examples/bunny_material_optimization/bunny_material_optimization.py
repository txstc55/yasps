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
from friction_helpers import closest_point_coord_and_tangent_basis_ee, closest_point_coord_and_tangent_basis_pe, closest_point_coord_and_tangent_basis_pp, closest_point_coord_and_tangent_basis_pt, friction_energy_ee, friction_energy_pe, friction_energy_pp, friction_energy_pt, lambda_last_h_ee, lambda_last_h_pe, lambda_last_h_pp, lambda_last_h_pt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "../ccd"))
from ccd import CCD

# Simulation, loss, material, solver, and optimizer settings.
OPTIMIZATION_TARGET = "mass-poisson-and-young"
OUTPUT_ROOT = SCRIPT_DIR

NUM_FRAMES = 200
NUM_ADJOINT_STEPS = 30
DT_VALUE = 0.01
DHAT_VALUE = 1e-6
KAPPA_VALUE = 10.0
FRICTION_RATE = 0.3
FLOOR_SIZE = 1000.0
FLOOR_HEIGHT = 0.0
FLOOR_CENTER = np.array([0.0, FLOOR_HEIGHT, 0.0], dtype=np.float64)
LOSS_TARGET = np.array([0.0, 2.0, 0.0], dtype=np.float64)
EVERY_FRAME_LOSS_WEIGHT_VALUE = 1.0
FINAL_FRAME_LOSS_WEIGHT_VALUE = 500.0
MOTION_LOSS_WEIGHT_VALUE = 30.0
DESIRED_DISPLACEMENT_VALUE = 0.2
MOTION_NORM_EPSILON = 1e-12
BUNNY_TOTAL_MASS = 10.0
MINIMUM_VERTEX_MASS_FRACTION = 0.1

POISSON_VALUE = 0.3045697005781997
YOUNG_VALUE = 15259.25455816859

SOLVER_TOLERANCE = 1e-6
ADJOINT_SOLVER_TOLERANCE = 1e-8
MOTION_TOLERANCE = 1e-2
MAX_NEWTON_ITERATIONS = 400
REFRESH_MAX_NEWTON_ITERATIONS = 400
REFRESH_NEWTON_EVERY = 2
MAX_CG_ITERATIONS = 20000
MAX_LINE_SEARCH_STEPS = 8
ADAM_LEARNING_RATE = 0.8
MASS_ADAM_LEARNING_RATE = 0.8
POISSON_STEP_SIZE = 0.9
YOUNG_STEP_SIZE = 0.9
POISSON_MIN_VALUE = 0.1
POISSON_MAX_VALUE = 0.49
YOUNG_MIN_VALUE = 500.0
YOUNG_MAX_VALUE = 500000.0
POISSON_LOGIT_DESIGN_LIMIT = 12.0
YOUNG_LOGIT_DESIGN_LIMIT = 12.0
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8
VIDEO_FPS = round(1.0 / DT_VALUE)


# Parse runtime, rendering, optimization, and checkpoint options.
parser = argparse.ArgumentParser(description="Drop one soft tetrahedral bunny onto a large static floor.")
parser.add_argument("--num-frames", type=int, default=NUM_FRAMES)
parser.add_argument("--adjoint-steps", type=int, default=NUM_ADJOINT_STEPS, help=("Exact number of backward passes, each followed by one forward pass " f"(default: {NUM_ADJOINT_STEPS})."),)
parser.add_argument("--adam-learning-rate", type=float, default=ADAM_LEARNING_RATE, help="Adam learning rate before applying the update-length cap.",)
parser.add_argument("--mass-adam-learning-rate", type=float, default=MASS_ADAM_LEARNING_RATE, help="Adam learning rate for the per-vertex log-mass design before fixed-total-mass normalization.")
parser.add_argument("--poisson-step-size", dest="poisson_step_size", type=float, default=POISSON_STEP_SIZE, help=("Maximum bounded-Poisson-logit update magnitude for any one element."),)
parser.add_argument("--young-step-size", type=float, default=YOUNG_STEP_SIZE, help="Maximum bounded-Young-logit update magnitude for any one element.",)
parser.add_argument("--preview-only", action="store_true", help="Show the initial PyVista scene without running the simulation.",)
parser.add_argument("--save-frames", action="store_true", help="Save one PNG per frame and encode the frames into an MP4.",)
parser.add_argument("--save-obj", action="store_true", help="Save only the bunny surface as one OBJ per frame.",)
parser.add_argument("--video-fps", type=int, default=VIDEO_FPS, help=("MP4 frame rate used with --save-frames " f"(default: {VIDEO_FPS}, matching dt={DT_VALUE})."),)
parser.add_argument("--no-gui", action="store_true", help="Run and render off screen without opening the interactive window.",)
parser.add_argument("--target-loss-final-frame-only", action="store_true", help=("Apply the target-position loss only at the final converged frame; " "the motion loss remains active at every frame."),)
parser.add_argument("--forward-only", choices=("initial", "best"), default=None, help="Run one forward trajectory without optimization, using either the initial design or the best design stored in --resume-checkpoint.")
resume_checkpoint_help = ("Resume an interrupted Adam optimization from latest_checkpoint.npz. " "The checkpoint design is replayed once to reconstruct its converged " "state trajectory before the next backward pass.")
parser.add_argument("--resume-checkpoint", type=str, default=None, help=resume_checkpoint_help)
parser.add_argument("--output-directory", type=str, default=None, help="Override the inverse-simulation output directory.")
args = parser.parse_args()

if args.num_frames < 0:
  raise ValueError("--num-frames must be non-negative.")
if args.adjoint_steps < 0:
  raise ValueError("--adjoint-steps must be non-negative.")
if args.poisson_step_size <= 0.0:
  raise ValueError("--poisson-step-size must be positive.")
if args.young_step_size <= 0.0:
  raise ValueError("--young-step-size must be positive.")
if args.adam_learning_rate <= 0.0:
  raise ValueError("--adam-learning-rate must be positive.")
if args.mass_adam_learning_rate <= 0.0:
  raise ValueError("--mass-adam-learning-rate must be positive.")
if args.video_fps <= 0:
  raise ValueError("--video-fps must be positive.")
if args.forward_only == "best" and args.resume_checkpoint is None:
  raise ValueError("--forward-only best requires --resume-checkpoint.")
loss_weight_value = FINAL_FRAME_LOSS_WEIGHT_VALUE if args.target_loss_final_frame_only else EVERY_FRAME_LOSS_WEIGHT_VALUE


# Small utilities for YASPS constants, mesh construction, statistics, and memory reporting.
def add_scalar_constant(owner, name, value):
  result = owner.addConstant(name, rows=1, cols=1)
  result.updateValue([value])
  return result


POISSON_BASE_FRACTION = (
  (POISSON_VALUE - POISSON_MIN_VALUE)
  / (POISSON_MAX_VALUE - POISSON_MIN_VALUE)
)
POISSON_BASE_LOGIT = np.log(POISSON_BASE_FRACTION / (1.0 - POISSON_BASE_FRACTION))
YOUNG_BASE_FRACTION = (
  (YOUNG_VALUE - YOUNG_MIN_VALUE)
  / (YOUNG_MAX_VALUE - YOUNG_MIN_VALUE)
)
YOUNG_BASE_LOGIT = np.log(YOUNG_BASE_FRACTION / (1.0 - YOUNG_BASE_FRACTION))


def poisson_values_from_design(design_parameters):
  design = np.asarray(design_parameters, dtype=np.float64).reshape(-1)
  shifted_logit = POISSON_BASE_LOGIT + design
  sigmoid = 1.0 / (1.0 + np.exp(-shifted_logit))
  poisson = (
    POISSON_MIN_VALUE
    + (POISSON_MAX_VALUE - POISSON_MIN_VALUE) * sigmoid
  )
  chain = (
    (poisson - POISSON_MIN_VALUE)
    * (POISSON_MAX_VALUE - poisson)
    / (POISSON_MAX_VALUE - POISSON_MIN_VALUE)
  )
  return poisson, chain


def young_values_from_design(design_parameters):
  design = np.asarray(design_parameters, dtype=np.float64).reshape(-1)
  shifted_logit = YOUNG_BASE_LOGIT + design
  sigmoid = 1.0 / (1.0 + np.exp(-shifted_logit))
  young_values = (
    YOUNG_MIN_VALUE
    + (YOUNG_MAX_VALUE - YOUNG_MIN_VALUE) * sigmoid
  )
  chain = (
    (young_values - YOUNG_MIN_VALUE)
    * (YOUNG_MAX_VALUE - young_values)
    / (YOUNG_MAX_VALUE - YOUNG_MIN_VALUE)
  )
  return young_values, chain


def material_values_from_parameters(young_values, poisson_values):
  young_values = np.asarray(young_values, dtype=np.float64).reshape(-1)
  poisson = np.asarray(poisson_values, dtype=np.float64).reshape(-1)
  if young_values.shape != poisson.shape:
    raise ValueError("Young and Poisson designs must have the same per-element layout.")
  return np.column_stack((young_values, poisson))


def material_statistics(material_values):
  values = np.asarray(material_values, dtype=np.float64).reshape((-1, 2))
  return {
    "young": {
      "min": float(values[:, 0].min()),
      "max": float(values[:, 0].max()),
      "mean": float(values[:, 0].mean()),
      "std": float(values[:, 0].std()),
    },
    "poisson": {
      "min": float(values[:, 1].min()),
      "max": float(values[:, 1].max()),
      "mean": float(values[:, 1].mean()),
      "std": float(values[:, 1].std()),
    },
  }


def format_material_statistics(statistics):
  young = statistics["young"]
  poisson = statistics["poisson"]
  return (
    f"young mean={young['mean']:.5g} "
    f"[{young['min']:.5g}, {young['max']:.5g}], "
    f"poisson mean={poisson['mean']:.5g} "
    f"[{poisson['min']:.5g}, {poisson['max']:.5g}]"
  )


def mass_statistics(vertex_masses):
  values = np.asarray(vertex_masses, dtype=np.float64).reshape(-1)
  return {
    "min": float(values.min()),
    "max": float(values.max()),
    "mean": float(values.mean()),
    "std": float(values.std()),
    "total": float(values.sum()),
  }


def format_mass_statistics(statistics):
  return (
    f"mass mean={statistics['mean']:.6e} "
    f"[{statistics['min']:.6e}, {statistics['max']:.6e}] "
    f"total={statistics['total']:.6e}"
  )


def maximum_element_norm(values):
  values = np.asarray(values, dtype=np.float64).reshape(-1)
  return float(np.max(np.abs(values), initial=0.0))


def mass_values_from_design(design_parameters, initial_vertex_masses):
  design = np.asarray(design_parameters, dtype=np.float64).reshape(-1)
  initial = np.broadcast_to(np.asarray(initial_vertex_masses, dtype=np.float64).reshape(-1), design.shape,)
  minimum = initial * MINIMUM_VERTEX_MASS_FRACTION
  free_mass_total = BUNNY_TOTAL_MASS - float(minimum.sum())
  weights = (initial - minimum) * np.exp(design - np.max(design))
  normalized_free_mass = free_mass_total * weights / float(weights.sum())
  values = minimum + normalized_free_mass
  if not np.all(np.isfinite(values)) or np.any(values < minimum) or not np.isclose(values.sum(), BUNNY_TOTAL_MASS, rtol=1e-12, atol=1e-10):
    raise RuntimeError("Normalized mass design violated its minimum, total-mass constraint, or finiteness.")
  return values, normalized_free_mass


def mass_design_gradient(value_gradient, normalized_free_mass):
  value_gradient = np.asarray(value_gradient, dtype=np.float64).reshape(-1)
  normalized_free_mass = np.asarray(normalized_free_mass, dtype=np.float64).reshape(-1)
  free_mass_total = float(normalized_free_mass.sum())
  weighted_mean_gradient = float(np.dot(value_gradient, normalized_free_mass)) / free_mass_total
  return normalized_free_mass * (value_gradient - weighted_mean_gradient)


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
  return positions, tetrahedra


def make_floor():
  half_size = 0.5 * FLOOR_SIZE
  positions = np.array([[-half_size, FLOOR_HEIGHT, -half_size], [half_size, FLOOR_HEIGHT, -half_size], [half_size, FLOOR_HEIGHT, half_size], [-half_size, FLOOR_HEIGHT, half_size],], dtype=np.float64)
  triangles = np.array([[0, 2, 1], [0, 3, 2],], dtype=np.uint32)
  return positions, triangles


def lump_tetrahedron_mass_to_vertices(positions, tetrahedra, total_mass):
  tetrahedron_positions = positions[tetrahedra]
  edge_matrices = np.stack([tetrahedron_positions[:, 1] - tetrahedron_positions[:, 0], tetrahedron_positions[:, 2] - tetrahedron_positions[:, 0], tetrahedron_positions[:, 3] - tetrahedron_positions[:, 0],], axis=1)
  tetrahedron_volumes = np.abs(np.linalg.det(edge_matrices)) / 6.0
  total_volume = float(tetrahedron_volumes.sum())
  if total_volume <= 0.0 or not np.isfinite(total_volume):
    raise RuntimeError("The bunny rest mesh has invalid total tetrahedral volume.")

  density = float(total_mass) / total_volume
  tetrahedron_masses = density * tetrahedron_volumes
  vertex_masses = np.zeros(positions.shape[0], dtype=np.float64)
  np.add.at(vertex_masses, tetrahedra.reshape(-1), np.repeat(0.25 * tetrahedron_masses, 4),)
  if np.any(vertex_masses <= 0.0) or not np.all(np.isfinite(vertex_masses)):
    raise RuntimeError("Volume lumping produced a non-positive vertex mass.")
  if not np.isclose(vertex_masses.sum(), total_mass, rtol=1e-12, atol=1e-12):
    raise RuntimeError("Volume-lumped vertex masses do not sum to the total mass.")
  return vertex_masses, total_volume, density


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
initial_vertex_masses, bunny_total_volume, bunny_density = lump_tetrahedron_mass_to_vertices(bunny_positions, bunny_tetrahedra, BUNNY_TOTAL_MASS)
minimum_vertex_masses = initial_vertex_masses * MINIMUM_VERTEX_MASS_FRACTION

# Build the YASPS scene and simulation state.

simulation = scene("bunny_material_optimization")
dt = add_scalar_constant(simulation, "dt", DT_VALUE)
dhat = add_scalar_constant(simulation, "dhat", DHAT_VALUE)
kappa = add_scalar_constant(simulation, "kappa", KAPPA_VALUE)
friction_rate = add_scalar_constant(simulation, "friction_rate", FRICTION_RATE)
loss_weight = add_scalar_constant(simulation, "loss_weight", loss_weight_value)
motion_loss_weight = add_scalar_constant(simulation, "motion_loss_weight", MOTION_LOSS_WEIGHT_VALUE)
desired_displacement = add_scalar_constant(simulation, "desired_displacement", DESIRED_DISPLACEMENT_VALUE)
motion_norm_epsilon = add_scalar_constant(simulation, "motion_norm_epsilon", MOTION_NORM_EPSILON)

bunny = simulation.addMesh("bunny_soft")

vertices_soft = bunny.addPrimitive("vertices_soft", numInstances=num_bunny_vertices)
position = vertices_soft.addAttribute("position", rows=3, cols=1)
rest_position = vertices_soft.addConstant("rest_position", rows=3, cols=1)
last_position = vertices_soft.addConstant("last_position", rows=3, cols=1)
last_last_position = vertices_soft.addConstant("last_last_position", rows=3, cols=1)
mass = vertices_soft.addConstant("mass", rows=1, cols=1)
velocity = (last_position - last_last_position) / dt

position.updateValue(bunny_positions.reshape(-1))
rest_position.updateValue(bunny_positions.reshape(-1))
last_position.updateValue(bunny_positions.reshape(-1))
last_last_position.updateValue(bunny_positions.reshape(-1))
mass.updateValue(initial_vertex_masses)

tets_soft = bunny.addPrimitive("tets_soft", numInstances=num_bunny_tetrahedra)
tets_to_vertices = tets_soft.addConnectivity("tets_to_vertices", vertices_soft, bunny_tetrahedra, 4)
tet_positions = tets_soft.addAttribute("positions", through=tets_to_vertices, source=position)
tet_rest_positions = tets_soft.addAttribute("rest_positions", through=tets_to_vertices, source=rest_position)
young = tets_soft.addConstant("young", rows=1, cols=1)
poisson = tets_soft.addConstant("poisson", rows=1, cols=1)
base_material_values = np.tile(np.array([YOUNG_VALUE, POISSON_VALUE], dtype=np.float64), (num_bunny_tetrahedra, 1),)
young.updateValue(base_material_values[:, 0])
poisson.updateValue(base_material_values[:, 1])
mu = young / (2.0 * (1.0 + poisson))
lam = (
  young * poisson
  / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
)

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
collision_last_position = collision_vertices.addAttribute("last_position")

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

pp_friction_pairs = collision_mesh.addPrimitive("pp_friction_pairs", numInstances=0, isDynamic=True)
pe_friction_pairs = collision_mesh.addPrimitive("pe_friction_pairs", numInstances=0, isDynamic=True)
pt_friction_pairs = collision_mesh.addPrimitive("pt_friction_pairs", numInstances=0, isDynamic=True)
ee_friction_pairs = collision_mesh.addPrimitive("ee_friction_pairs", numInstances=0, isDynamic=True)
pp_friction_to_vertices = pp_friction_pairs.addConnectivity("pp_friction_to_vertices", collision_vertices, [], 2)
pe_friction_to_vertices = pe_friction_pairs.addConnectivity("pe_friction_to_vertices", collision_vertices, [], 3)
pt_friction_to_vertices = pt_friction_pairs.addConnectivity("pt_friction_to_vertices", collision_vertices, [], 4)
ee_friction_to_vertices = ee_friction_pairs.addConnectivity("ee_friction_to_vertices", collision_vertices, [], 4)
pp_friction_positions = pp_friction_pairs.addAttribute("positions", through=pp_friction_to_vertices, source=collision_position)
pe_friction_positions = pe_friction_pairs.addAttribute("positions", through=pe_friction_to_vertices, source=collision_position)
pt_friction_positions = pt_friction_pairs.addAttribute("positions", through=pt_friction_to_vertices, source=collision_position)
ee_friction_positions = ee_friction_pairs.addAttribute("positions", through=ee_friction_to_vertices, source=collision_position)
pp_friction_last_positions = pp_friction_pairs.addAttribute("last_positions", through=pp_friction_to_vertices, source=collision_last_position)
pe_friction_last_positions = pe_friction_pairs.addAttribute("last_positions", through=pe_friction_to_vertices, source=collision_last_position)
pt_friction_last_positions = pt_friction_pairs.addAttribute("last_positions", through=pt_friction_to_vertices, source=collision_last_position)
ee_friction_last_positions = ee_friction_pairs.addAttribute("last_positions", through=ee_friction_to_vertices, source=collision_last_position)

pp_friction_coord, pp_friction_tangent_basis = closest_point_coord_and_tangent_basis_pp(pp_friction_last_positions)
pe_friction_coord, pe_friction_tangent_basis = closest_point_coord_and_tangent_basis_pe(pe_friction_last_positions)
pt_friction_coord, pt_friction_tangent_basis = closest_point_coord_and_tangent_basis_pt(pt_friction_last_positions)
ee_friction_coord, ee_friction_tangent_basis = closest_point_coord_and_tangent_basis_ee(ee_friction_last_positions)
pp_friction_pairs.addAttribute("coord", computed_attribute=pp_friction_coord)
pe_friction_pairs.addAttribute("coord", computed_attribute=pe_friction_coord)
pt_friction_pairs.addAttribute("coord", computed_attribute=pt_friction_coord)
ee_friction_pairs.addAttribute("coord", computed_attribute=ee_friction_coord)
pp_friction_pairs.addAttribute("tangent_basis", computed_attribute=pp_friction_tangent_basis)
pe_friction_pairs.addAttribute("tangent_basis", computed_attribute=pe_friction_tangent_basis)
pt_friction_pairs.addAttribute("tangent_basis", computed_attribute=pt_friction_tangent_basis)
ee_friction_pairs.addAttribute("tangent_basis", computed_attribute=ee_friction_tangent_basis)

pp_friction_lambda_last_h = lambda_last_h_pp(pp_friction_last_positions, pp_friction_coord, dhat, kappa)
pe_friction_lambda_last_h = lambda_last_h_pe(pe_friction_last_positions, pe_friction_coord, dhat, kappa)
pt_friction_lambda_last_h = lambda_last_h_pt(pt_friction_last_positions, pt_friction_coord, dhat, kappa)
ee_friction_lambda_last_h = lambda_last_h_ee(ee_friction_last_positions, ee_friction_coord, dhat, kappa)
pp_friction_pairs.addAttribute("lambda_last_h", computed_attribute=pp_friction_lambda_last_h)
pe_friction_pairs.addAttribute("lambda_last_h", computed_attribute=pe_friction_lambda_last_h)
pt_friction_pairs.addAttribute("lambda_last_h", computed_attribute=pt_friction_lambda_last_h)
ee_friction_pairs.addAttribute("lambda_last_h", computed_attribute=ee_friction_lambda_last_h)



# Define inertia, elasticity, contact, friction, and loss energies.

elastic_energy = tets_soft.addAttribute("elastic_energy", computed_attribute=stable_neo_hookean(tet_rest_positions, tet_positions, mu, lam, dt))
inertia_energy = vertices_soft.addAttribute("inertia_energy", computed_attribute=inertia(last_position, velocity, dt, position, mass))
pp_energy = collision_mesh.pp.addAttribute("point_point_energy", computed_attribute=point_point(pp_positions, dhat, kappa))
pe_energy = collision_mesh.pe.addAttribute("point_edge_energy", computed_attribute=point_edge(pe_positions, dhat, kappa))
pt_energy = collision_mesh.pt.addAttribute("point_triangle_energy", computed_attribute=point_triangle(pt_positions, dhat, kappa))
ee_energy = collision_mesh.ee.addAttribute("edge_edge_energy", computed_attribute=edge_edge(ee_positions, dhat, kappa))
pp_friction_energy = pp_friction_pairs.addAttribute("friction_energy", computed_attribute=friction_energy_pp(pp_friction_positions, pp_friction_last_positions, dhat, dt, friction_rate, pp_friction_coord, pp_friction_tangent_basis.row(0), pp_friction_tangent_basis.row(1), pp_friction_lambda_last_h))
pe_friction_energy = pe_friction_pairs.addAttribute("friction_energy", computed_attribute=friction_energy_pe(pe_friction_positions, pe_friction_last_positions, dhat, dt, friction_rate, pe_friction_coord, pe_friction_tangent_basis.row(0), pe_friction_tangent_basis.row(1), pe_friction_lambda_last_h))
pt_friction_energy = pt_friction_pairs.addAttribute("friction_energy", computed_attribute=friction_energy_pt(pt_friction_positions, pt_friction_last_positions, dhat, dt, friction_rate, pt_friction_coord, pt_friction_tangent_basis.row(0), pt_friction_tangent_basis.row(1), pt_friction_lambda_last_h))
ee_friction_energy = ee_friction_pairs.addAttribute("friction_energy", computed_attribute=friction_energy_ee(ee_friction_positions, ee_friction_last_positions, dhat, dt, friction_rate, ee_friction_coord, ee_friction_tangent_basis.row(0), ee_friction_tangent_basis.row(1), ee_friction_lambda_last_h))
loss_offset = position - attribute.to_array(LOSS_TARGET.tolist(), rows=3, cols=1)
target_position_loss = vertices_soft.addAttribute("target_position_loss", computed_attribute=(loss_offset.dot(loss_offset)) * loss_weight)
frame_displacement = position - last_position
frame_displacement_magnitude = (frame_displacement.dot(frame_displacement) + motion_norm_epsilon).sqrt()
motion_error = frame_displacement_magnitude - desired_displacement
motion_reward_loss = vertices_soft.addAttribute("motion_reward_loss", computed_attribute=motion_loss_weight * motion_error * motion_error)
target_state_loss_gradient = differentiator().diff1([target_position_loss], [position])
motion_state_loss_gradient = differentiator().diff1([motion_reward_loss], [position])
previous_state_loss_gradient = differentiator().diff1([motion_reward_loss], [last_position])

simulation.addEnergy(elastic_energy, projection_method=1)
simulation.addEnergy(inertia_energy, projection_method=-1)
simulation.addEnergy(pp_energy, dynamic_instances=True, projection_method=2)
simulation.addEnergy(pe_energy, dynamic_instances=True, projection_method=2)
simulation.addEnergy(pt_energy, dynamic_instances=True, projection_method=2)
simulation.addEnergy(ee_energy, dynamic_instances=True, projection_method=2)
simulation.addEnergy(pp_friction_energy, dynamic_instances=True, projection_method=2)
simulation.addEnergy(pe_friction_energy, dynamic_instances=True, projection_method=2)
simulation.addEnergy(pt_friction_energy, dynamic_instances=True, projection_method=2)
simulation.addEnergy(ee_friction_energy, dynamic_instances=True, projection_method=2)
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


def restore_friction_pairs(saved_pairs):
  pp, pe, pt, ee = saved_pairs
  pp_count = pp.size // 2
  pe_count = pe.size // 3
  pt_count = pt.size // 4
  ee_count = ee.size // 4

  pp_friction_pairs.updateNumInstances(pp_count)
  pe_friction_pairs.updateNumInstances(pe_count)
  pt_friction_pairs.updateNumInstances(pt_count)
  ee_friction_pairs.updateNumInstances(ee_count)

  if pp_count > 0:
    pp_friction_to_vertices.updateConnectivity(pp.reshape((-1, 2)))
  if pe_count > 0:
    pe_friction_to_vertices.updateConnectivity(pe.reshape((-1, 3)))
  if pt_count > 0:
    pt_friction_to_vertices.updateConnectivity(pt.reshape((-1, 4)))
  if ee_count > 0:
    ee_friction_to_vertices.updateConnectivity(ee.reshape((-1, 4)))


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


def update_friction_pairs():
  pp_count, pe_count, pt_count, ee_count = (
    int(value) for value in ccd.separated_counts
  )
  saved_pairs = (
    ccd.pp[:2 * pp_count].get().copy(),
    ccd.pe[:3 * pe_count].get().copy(),
    ccd.pt[:4 * pt_count].get().copy(),
    ccd.ee[:4 * ee_count].get().copy(),
  )
  restore_friction_pairs(saved_pairs)
  return saved_pairs

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
  differentiator().diff2([pp_friction_energy], [position], [position], projection_method=2, dynamic_instances=True),
  differentiator().diff2([pe_friction_energy], [position], [position], projection_method=2, dynamic_instances=True),
  differentiator().diff2([pt_friction_energy], [position], [position], projection_method=2, dynamic_instances=True),
  differentiator().diff2([ee_friction_energy], [position], [position], projection_method=2, dynamic_instances=True),
]
adjoint_hessian = add_matrices(adjoint_hessian_terms)

# Build residual Jacobians used to propagate adjoints and design gradients.

previous_position_jacobian_terms = [
  differentiator().diff2([inertia_energy], [position], [last_position]),
  differentiator().diff2([pp_friction_energy], [position], [last_position], dynamic_instances=True),
  differentiator().diff2([pe_friction_energy], [position], [last_position], dynamic_instances=True),
  differentiator().diff2([pt_friction_energy], [position], [last_position], dynamic_instances=True),
  differentiator().diff2([ee_friction_energy], [position], [last_position], dynamic_instances=True),
]
previous_position_jacobian = add_matrices(previous_position_jacobian_terms)
previous_previous_position_jacobian = differentiator().diff2([inertia_energy], [position], [last_last_position])
poisson_jacobian = differentiator().diff2([elastic_energy], [position], [poisson])
young_jacobian = differentiator().diff2([elastic_energy], [position], [young])
mass_jacobian = differentiator().diff2([inertia_energy], [position], [mass])
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
bunny_surface_local_indices = np.full(num_bunny_vertices, -1, dtype=np.int64)
bunny_surface_local_indices[bunny_surface_indices] = np.arange(bunny_surface_indices.size)
bunny_surface_triangles_local = bunny_surface_local_indices[bunny_surface_triangles]
if np.any(bunny_surface_triangles_local < 0):
  raise RuntimeError("A bunny surface triangle references a non-surface vertex.")
bunny_surface_triangles_local = bunny_surface_triangles_local.astype(np.uint32)
bunny_cells = np.hstack([np.full((bunny_surface_triangles_local.shape[0], 1), 3, dtype=np.uint32), bunny_surface_triangles_local])
floor_cells = np.hstack([np.full((floor_triangles.shape[0], 1), 3, dtype=np.uint32), floor_triangles])

bunny_poly = pv.PolyData(bunny_positions[bunny_surface_indices], bunny_cells)
floor_poly = pv.PolyData(floor_positions, floor_cells)

plotter = pv.Plotter(window_size=[1920, 1080], off_screen=args.no_gui)
plotter.add_mesh(bunny_poly, color="lightgreen", opacity=0.65, smooth_shading=True, show_edges=False)
plotter.add_mesh(floor_poly, color="lightgray", opacity=0.25, show_edges=True)
floor_center_marker = pv.Sphere(radius=0.05, center=FLOOR_CENTER, theta_resolution=32, phi_resolution=32)
plotter.add_mesh(floor_center_marker, color="red", ambient=1.0, pickable=False,)
loss_target_marker = pv.Sphere(radius=0.08, center=LOSS_TARGET, theta_resolution=32, phi_resolution=32)
plotter.add_mesh(loss_target_marker, color="blue", ambient=1.0, pickable=False,)
plotter.add_text("Initial configuration", name="optimization_status", position="upper_left", font_size=16, color="black")
plotter.camera_position = [
  (5.0, 3.5, 9.0),
  (0.0, 2.25, 0.0),
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

default_output_name = f"inverse_{OPTIMIZATION_TARGET.replace('-', '_')}_projected_adam"
output_directory = args.output_directory or os.path.join(OUTPUT_ROOT, "outputs", default_output_name,)
output_directory = os.path.abspath(output_directory)
os.makedirs(output_directory, exist_ok=True)


def refresh_gui():
  current_positions = position.compute().value.get().reshape((-1, 3))
  bunny_poly.points = current_positions[bunny_surface_indices]
  plotter.render()
  if not args.no_gui:
    plotter.update()
  return current_positions


# Export the current surface-only bunny directly through PyVista's OBJ writer.
def save_bunny_obj(obj_directory, frame):
  obj_path = os.path.join(obj_directory, f"bunny_{frame:04d}.obj")
  pv.PolyData(bunny_poly.points, bunny_poly.faces).save(obj_path)
  return obj_path


def output_paths(adjoint_round=None, run_name=None):
  round_name = run_name if run_name is not None else ("baseline" if adjoint_round == 0 else f"adjoint_{adjoint_round:02d}")
  round_directory = os.path.join(output_directory, round_name)
  return (
    round_name,
    os.path.join(round_directory, "frames"),
    os.path.join(round_directory, "bunny_obj"),
    os.path.join(output_directory, f"{round_name}.mp4"),
  )


def encode_saved_frames(adjoint_round, loss, baseline_loss=None, run_name=None):
  round_name, frame_directory, _, video_path = output_paths(adjoint_round, run_name)
  ffmpeg = shutil.which("ffmpeg")
  if ffmpeg is None:
    raise RuntimeError("--save-frames requires ffmpeg to create the final MP4.")

  if run_name is not None:
    loss_text = f"{run_name.replace('_', ' ')}   loss {loss:.8f}"
  elif adjoint_round == 0:
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
def run_forward(young_design, poisson_design, mass_design, adjoint_round=None, baseline_loss=None, run_name=None):
  forward_index = 0 if adjoint_round is None else adjoint_round
  max_newton_iterations = (
    REFRESH_MAX_NEWTON_ITERATIONS
    if forward_index % REFRESH_NEWTON_EVERY == 0
    else MAX_NEWTON_ITERATIONS
  )
  poisson_values, poisson_chain = poisson_values_from_design(poisson_design)
  young_values, young_chain = young_values_from_design(young_design)
  young.updateValue(young_values)
  poisson.updateValue(poisson_values)
  material_values = material_values_from_parameters(young_values, poisson_values,)
  material_summary = material_statistics(material_values)
  vertex_masses, mass_chain = mass_values_from_design(mass_design, initial_vertex_masses,)
  mass.updateValue(vertex_masses)
  vertex_mass_summary = mass_statistics(vertex_masses)
  initial_position = bunny_positions.copy()
  position.updateValue(initial_position.reshape(-1))
  last_position.updateValue(initial_position.reshape(-1))
  last_last_position.updateValue(initial_position.reshape(-1))
  update_collision_pairs()
  refresh_gui()

  record_adjoint_state = run_name is None
  previous_converged_positions = initial_position.copy()
  saved_positions = [initial_position.copy()] if record_adjoint_state else []
  saved_collision_pairs = []
  saved_friction_pairs = []
  frame_losses = []
  target_frame_losses = []
  motion_frame_losses = []
  mean_step_displacements = []
  maximum_step_displacements = []
  mean_vertical_step_displacements = []
  maximum_vertical_step_displacements = []
  floor_minimum_y_values = []
  floor_penetrating_vertex_counts = []
  floor_maximum_penetrations = []
  floor_minimum_surface_y_values = []
  floor_penetrating_surface_vertex_counts = []
  floor_maximum_surface_penetrations = []
  centroid_y_values = [float(initial_position[:, 1].mean())]
  loss = 0.0
  target_loss = 0.0
  motion_loss = 0.0
  saved_video = None
  obj_directory = None

  save_this_forward = adjoint_round is not None or run_name is not None
  forward_label = run_name if run_name is not None else ("initial" if adjoint_round in (None, 0) else f"adjoint={adjoint_round:02d}")
  if save_this_forward:
    _, frame_directory, default_obj_directory, _ = output_paths(adjoint_round, run_name)
    design_status = (
      f"{format_material_statistics(material_summary)}\n"
      f"{format_mass_statistics(vertex_mass_summary)}"
    )
    status_title = run_name.replace("_", " ").title() if run_name is not None else ("Unoptimized baseline" if adjoint_round == 0 else f"After adjoint {adjoint_round:02d}/{args.adjoint_steps:02d}")
    plotter.add_text(f"{status_title}\n{design_status}", name="optimization_status", position="upper_left", font_size=16, color="black")
    if args.save_frames:
      os.makedirs(frame_directory, exist_ok=True)
    if args.save_obj:
      if run_name is not None:
        obj_directory = default_obj_directory
      elif adjoint_round == 0:
        obj_directory = os.path.join(output_directory, "baseline", "bunny_obj")
      else:
        candidate_directory = os.path.join(output_directory, "_candidate_best")
        if os.path.isdir(candidate_directory):
          shutil.rmtree(candidate_directory)
        obj_directory = os.path.join(candidate_directory, "bunny_obj")
      os.makedirs(obj_directory, exist_ok=True)

  collision_position_copy = collision_position.compute().value.copy()
  bunny_position_copy = position.compute().value.copy()
  gpu_start_used, gpu_total = gpu_memory_used()
  gpu_peak_used = gpu_start_used
  forward_start = time.time()

  for frame in range(args.num_frames):
    last_last_position.updateValue(last_position.value, deepCopy=True)
    last_position.updateValue(position.value, deepCopy=True)
    update_collision_pairs()
    frame_friction_pairs = update_friction_pairs()

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

      prefix = forward_label
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
    bunny_y = current_positions[:, 1]
    minimum_y = float(bunny_y.min())
    penetrating_vertex_count = int(np.count_nonzero(bunny_y < FLOOR_HEIGHT))
    maximum_penetration = max(0.0, FLOOR_HEIGHT - minimum_y)
    bunny_surface_y = bunny_y[bunny_surface_indices]
    minimum_surface_y = float(bunny_surface_y.min())
    penetrating_surface_vertex_count = int(np.count_nonzero(bunny_surface_y < FLOOR_HEIGHT))
    maximum_surface_penetration = max(0.0, FLOOR_HEIGHT - minimum_surface_y)
    penetrating_vertex_indices = np.flatnonzero(bunny_y < FLOOR_HEIGHT).tolist()
    floor_minimum_y_values.append(minimum_y)
    floor_penetrating_vertex_counts.append(penetrating_vertex_count)
    floor_maximum_penetrations.append(maximum_penetration)
    floor_minimum_surface_y_values.append(minimum_surface_y)
    floor_penetrating_surface_vertex_counts.append(penetrating_surface_vertex_count)
    floor_maximum_surface_penetrations.append(maximum_surface_penetration)
    frame_prefix = forward_label
    floor_check_message = (
      f"floor_check {frame_prefix} frame={frame:03d} "
      f"minimum_y={minimum_y:.12e} "
      f"vertices_below_floor={penetrating_vertex_count} "
      f"maximum_penetration={maximum_penetration:.12e} "
      f"minimum_surface_y={minimum_surface_y:.12e} "
      f"surface_vertices_below_floor={penetrating_surface_vertex_count} "
      f"maximum_surface_penetration={maximum_surface_penetration:.12e} "
      f"penetrating_vertex_indices_sample={penetrating_vertex_indices[:16]}"
    )
    print(floor_check_message)
    target_loss_is_active = (
      not args.target_loss_final_frame_only
      or frame == args.num_frames - 1
    )
    target_frame_loss = float(target_position_loss.compute().value.get().sum()) if target_loss_is_active else 0.0
    motion_frame_loss = float(motion_reward_loss.compute().value.get().sum())
    frame_loss = target_frame_loss + motion_frame_loss
    frame_losses.append(frame_loss)
    target_frame_losses.append(target_frame_loss)
    motion_frame_losses.append(motion_frame_loss)
    loss += frame_loss
    target_loss += target_frame_loss
    motion_loss += motion_frame_loss
    step_displacements = np.linalg.norm(current_positions - previous_converged_positions, axis=1,)
    mean_step_displacements.append(float(step_displacements.mean()))
    maximum_step_displacements.append(float(step_displacements.max()))
    vertical_step_displacements = np.abs(current_positions[:, 1] - previous_converged_positions[:, 1])
    mean_vertical_step_displacements.append(float(vertical_step_displacements.mean()))
    maximum_vertical_step_displacements.append(float(vertical_step_displacements.max()))
    centroid_y_values.append(float(current_positions[:, 1].mean()))

    # Only converged state data are checkpointed.  Numeric Hessian and
    # Jacobian values are deliberately not stored; the reverse loop restores
    # these inputs and recomputes all matrices on demand.
    if record_adjoint_state:
      saved_positions.append(current_positions.copy())
      saved_collision_pairs.append(save_collision_pairs())
      saved_friction_pairs.append(frame_friction_pairs)
    previous_converged_positions = current_positions.copy()

    if save_this_forward and args.save_frames:
      plotter.screenshot(os.path.join(frame_directory, f"frame_{frame:04d}.png",))
    if save_this_forward and args.save_obj:
      save_bunny_obj(obj_directory, frame)

  final_positions = position.compute().value.get().reshape((-1, 3))
  terminal_position_summary = {
    "centroid": final_positions.mean(axis=0).tolist(),
    "minimum_y": float(final_positions[:, 1].min()),
    "maximum_y": float(final_positions[:, 1].max()),
  }
  frame_loss_summary = {
    "minimum": min(frame_losses) if frame_losses else 0.0,
    "maximum": max(frame_losses) if frame_losses else 0.0,
    "sum": loss,
    "target_sum": target_loss,
    "motion_reward_sum": motion_loss,
  }
  apex_state_index = int(np.argmax(centroid_y_values))
  motion_summary = {
    "mean_step_displacement": float(np.mean(mean_step_displacements)) if mean_step_displacements else 0.0,
    "maximum_mean_step_displacement": max(mean_step_displacements) if mean_step_displacements else 0.0,
    "maximum_vertex_step_displacement": max(maximum_step_displacements) if maximum_step_displacements else 0.0,
    "maximum_mean_speed": max(mean_step_displacements) / DT_VALUE if mean_step_displacements else 0.0,
    "mean_vertical_step_displacement": float(np.mean(mean_vertical_step_displacements)) if mean_vertical_step_displacements else 0.0,
    "maximum_mean_vertical_step_displacement": max(mean_vertical_step_displacements) if mean_vertical_step_displacements else 0.0,
    "maximum_vertex_vertical_step_displacement": max(maximum_vertical_step_displacements) if maximum_vertical_step_displacements else 0.0,
    "maximum_mean_vertical_speed": max(mean_vertical_step_displacements) / DT_VALUE if mean_vertical_step_displacements else 0.0,
  }
  floor_collision_summary = {
    "floor_height": FLOOR_HEIGHT,
    "minimum_y": min(floor_minimum_y_values) if floor_minimum_y_values else float(initial_position[:, 1].min()),
    "frames_with_vertices_below_floor": int(np.count_nonzero(np.asarray(floor_penetrating_vertex_counts) > 0)),
    "maximum_vertices_below_floor": max(floor_penetrating_vertex_counts) if floor_penetrating_vertex_counts else 0,
    "maximum_penetration": max(floor_maximum_penetrations) if floor_maximum_penetrations else 0.0,
    "minimum_surface_y": min(floor_minimum_surface_y_values) if floor_minimum_surface_y_values else float(initial_position[bunny_surface_indices, 1].min()),
    "frames_with_surface_vertices_below_floor": int(np.count_nonzero(np.asarray(floor_penetrating_surface_vertex_counts) > 0)),
    "maximum_surface_vertices_below_floor": max(floor_penetrating_surface_vertex_counts) if floor_penetrating_surface_vertex_counts else 0,
    "maximum_surface_penetration": max(floor_maximum_surface_penetrations) if floor_maximum_surface_penetrations else 0.0,
  }
  centroid_height_summary = {
    "initial": centroid_y_values[0],
    "minimum": min(centroid_y_values),
    "maximum": centroid_y_values[apex_state_index],
    "apex_state_index": apex_state_index,
    "apex_time": apex_state_index * DT_VALUE,
    "final": centroid_y_values[-1],
  }
  gpu_end_used, _ = gpu_memory_used()
  gpu_peak_used = max(gpu_peak_used, gpu_end_used)
  memory = gpu_memory_summary(gpu_start_used, gpu_peak_used, gpu_end_used, gpu_total)
  elapsed = time.time() - forward_start
  forward_summary_message = (
    f"Finished forward "
    f"{forward_label} in "
    f"{elapsed:.2f}s: loss={loss:.8e} "
    f"(target={target_loss:.8e}, motion={motion_loss:.8e}), "
    f"centroid_height={centroid_height_summary}, "
    f"motion={motion_summary}, "
    f"floor_collision={floor_collision_summary}, "
    f"newton_cap={max_newton_iterations}, gpu_memory={memory}."
  )
  print(forward_summary_message)

  if save_this_forward and args.save_frames:
    saved_video = encode_saved_frames(adjoint_round, loss, baseline_loss=baseline_loss, run_name=run_name)

  return {
    "positions": saved_positions,
    "collision_pairs": saved_collision_pairs,
    "friction_pairs": saved_friction_pairs,
    "loss": loss,
    "frame_loss_summary": frame_loss_summary,
    "motion_summary": motion_summary,
    "floor_collision_summary": floor_collision_summary,
    "centroid_height_summary": centroid_height_summary,
    "elapsed_seconds": elapsed,
    "max_newton_iterations": max_newton_iterations,
    "young_values": young_values,
    "young_chain": young_chain,
    "poisson_values": poisson_values,
    "poisson_chain": poisson_chain,
    "material_values": material_values,
    "material_summary": material_summary,
    "vertex_masses": vertex_masses.copy(),
    "mass_chain": mass_chain,
    "mass_design": mass_design.copy(),
    "mass_summary": vertex_mass_summary,
    "terminal_position_summary": terminal_position_summary,
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
    vector(motion_state_loss_gradient.size)
    for _ in range(args.num_frames + 2)
  ]
  young_gradient = vector(young_jacobian.cols)
  poisson_gradient = vector(poisson_jacobian.cols)
  mass_gradient = vector(mass_jacobian.cols)
  state_loss_gradient_inf_min = float("inf")
  state_loss_gradient_inf_max = 0.0
  previous_state_loss_gradient_inf_min = float("inf")
  previous_state_loss_gradient_inf_max = 0.0
  gpu_start_used, gpu_total = gpu_memory_used()
  gpu_peak_used = gpu_start_used
  backward_start = time.time()
  timing = {
    "state_restore_seconds": 0.0,
    "state_loss_gradient_seconds": 0.0,
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
    restore_friction_pairs(trajectory["friction_pairs"][frame])
    timing["state_restore_seconds"] += time.time() - restore_start

    # The motion reward contributes at every frame to q[k + 1] and q[k].
    # The target term contributes to q[k + 1] either at every frame or only
    # at the final frame, according to the selected loss schedule.
    loss_gradient_start = time.time()
    motion_state_loss_gradient.compute()
    previous_state_loss_gradient.compute()
    current_state_loss_gradient = vector(motion_state_loss_gradient.size)
    current_state_loss_gradient.updateValue(motion_state_loss_gradient)
    if not args.target_loss_final_frame_only or frame == args.num_frames - 1:
      target_state_loss_gradient.compute()
      current_target_state_loss_gradient = vector(target_state_loss_gradient.size)
      current_target_state_loss_gradient.updateValue(target_state_loss_gradient)
      current_state_loss_gradient = current_state_loss_gradient + current_target_state_loss_gradient
    current_previous_state_loss_gradient = vector(previous_state_loss_gradient.size)
    current_previous_state_loss_gradient.updateValue(previous_state_loss_gradient)
    current_state_loss_gradient_inf = float(gpuarray.max(abs(current_state_loss_gradient.value)).get())
    current_previous_state_loss_gradient_inf = float(gpuarray.max(abs(current_previous_state_loss_gradient.value)).get())
    state_loss_gradient_inf_min = min(state_loss_gradient_inf_min, current_state_loss_gradient_inf,)
    state_loss_gradient_inf_max = max(state_loss_gradient_inf_max, current_state_loss_gradient_inf,)
    previous_state_loss_gradient_inf_min = min(previous_state_loss_gradient_inf_min, current_previous_state_loss_gradient_inf,)
    previous_state_loss_gradient_inf_max = max(previous_state_loss_gradient_inf_max, current_previous_state_loss_gradient_inf,)
    position_adjoints[frame + 2] = (
      position_adjoints[frame + 2] + current_state_loss_gradient
    )
    position_adjoints[frame + 1] = (
      position_adjoints[frame + 1]
      + current_previous_state_loss_gradient
    )
    timing["state_loss_gradient_seconds"] += (
      time.time() - loss_gradient_start
    )

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
    poisson_jacobian.compute()
    poisson_gradient = poisson_gradient - poisson_jacobian.spmv(auxiliary, transpose=True)
    young_jacobian.compute()
    young_gradient = young_gradient - young_jacobian.spmv(auxiliary, transpose=True)
    mass_jacobian.compute()
    mass_gradient = mass_gradient - mass_jacobian.spmv(auxiliary, transpose=True)
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
      adjoint_frame_message = (
        f"adjoint={adjoint_round:02d} reverse_frame={frame:03d} "
        f"|g_current|_inf={current_state_loss_gradient_inf:.6e} "
        f"|g_previous|_inf={current_previous_state_loss_gradient_inf:.6e} "
        f"|lambda_x|_inf={float(gpuarray.max(abs(position_adjoints[frame + 1].value)).get()):.6e} "
        "solver=gpu_cg"
      )
      print(adjoint_frame_message)

  young_gradient_values = young_gradient.value.get()
  poisson_gradient_values = poisson_gradient.value.get()
  mass_gradient_values = mass_gradient.value.get()
  gradient_summary = (
    "young_gradient "
    f"l2={np.linalg.norm(young_gradient_values):.6e} "
    f"inf={np.max(np.abs(young_gradient_values)):.6e}, "
    "poisson_gradient "
    f"l2={np.linalg.norm(poisson_gradient_values):.6e} "
    f"inf={np.max(np.abs(poisson_gradient_values)):.6e}, "
    f"mass_gradient_l2={np.linalg.norm(mass_gradient_values):.6e} "
    f"mass_gradient_inf={np.max(np.abs(mass_gradient_values)):.6e}"
  )
  elapsed = time.time() - backward_start
  gpu_end_used, _ = gpu_memory_used()
  gpu_peak_used = max(gpu_peak_used, gpu_end_used)
  memory = gpu_memory_summary(gpu_start_used, gpu_peak_used, gpu_end_used, gpu_total,)
  timing["other_seconds"] = elapsed - sum(timing.values())
  if not np.isfinite(state_loss_gradient_inf_min):
    state_loss_gradient_inf_min = 0.0
  if not np.isfinite(previous_state_loss_gradient_inf_min):
    previous_state_loss_gradient_inf_min = 0.0
  adjoint_summary_message = (
    f"Finished adjoint {adjoint_round:02d} in {elapsed:.2f}s: "
    f"{gradient_summary}, "
    f"state_loss_gradient_inf_range="
    f"[{state_loss_gradient_inf_min:.6e}, {state_loss_gradient_inf_max:.6e}], "
    f"previous_state_loss_gradient_inf_range="
    f"[{previous_state_loss_gradient_inf_min:.6e}, {previous_state_loss_gradient_inf_max:.6e}], "
    f"timing={timing}, gpu_memory={memory}."
  )
  print(adjoint_summary_message)
  return (
    young_gradient_values,
    poisson_gradient_values,
    mass_gradient_values,
    elapsed,
    timing,
    memory,
  )


# Run a baseline or saved-best trajectory without retaining adjoint checkpoints.
def run_forward_only():
  young_design = np.zeros(num_bunny_tetrahedra, dtype=np.float64)
  poisson_design = np.zeros(num_bunny_tetrahedra, dtype=np.float64)
  mass_design = np.zeros(num_bunny_vertices, dtype=np.float64)
  best_round = None
  checkpoint_best_loss = None
  run_name = "initial"
  if args.forward_only == "best":
    checkpoint_path = os.path.abspath(args.resume_checkpoint)
    if not os.path.isfile(checkpoint_path):
      raise FileNotFoundError(f"Forward checkpoint does not exist: {checkpoint_path}")
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
      required_values = {"best_adjoint_round", "best_loss", "best_young_design", "best_poisson_design", "best_mass_design"}
      missing_values = required_values.difference(checkpoint.files)
      if missing_values:
        raise ValueError(f"Forward checkpoint is missing: {sorted(missing_values)}")
      best_round = int(checkpoint["best_adjoint_round"].item())
      checkpoint_best_loss = float(checkpoint["best_loss"].item())
      young_design = checkpoint["best_young_design"].copy()
      poisson_design = checkpoint["best_poisson_design"].copy()
      mass_design = checkpoint["best_mass_design"].copy()
    expected_shapes = ((young_design, (num_bunny_tetrahedra,)), (poisson_design, (num_bunny_tetrahedra,)), (mass_design, (num_bunny_vertices,)))
    if any(value.shape != expected_shape for value, expected_shape in expected_shapes):
      raise ValueError("The saved best design dimensions do not match the current bunny mesh.")
    run_name = f"best_round_{best_round:02d}"
  trajectory = run_forward(young_design, poisson_design, mass_design, run_name=run_name)
  _, _, obj_directory, _ = output_paths(run_name=run_name)
  summary = {"run_name": run_name, "num_frames": args.num_frames, "checkpoint_best_round": best_round, "checkpoint_best_loss": checkpoint_best_loss, "recomputed_loss": trajectory["loss"], "elapsed_seconds": trajectory["elapsed_seconds"], "gpu_memory": trajectory["gpu_memory"], "material_summary": trajectory["material_summary"], "mass_summary": trajectory["mass_summary"], "terminal_position_summary": trajectory["terminal_position_summary"], "floor_collision_summary": trajectory["floor_collision_summary"], "obj_directory": obj_directory if args.save_obj else None}
  summary_path = os.path.join(output_directory, f"{run_name}_summary.json")
  with open(summary_path, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2)
  print(f"Finished forward-only run {run_name}: loss={trajectory['loss']:.8e}, elapsed={trajectory['elapsed_seconds']:.2f}s, objs={summary['obj_directory']}.")
  print(f"Saved forward-only summary: {summary_path}")
  plotter.close()


update_collision_pairs()
refresh_gui()

if args.forward_only is not None:
  run_forward_only()
  raise SystemExit(0)

# Initialize or restore the design, then alternate one backward and one forward solve.
optimization_start = time.time()
optimization_history = []
resume_completed_adjoint_round = 0
resumed_from_checkpoint = None
resume_reconstructed_loss = None
resume_reconstructed_loss_relative_difference = None

if args.resume_checkpoint is None:
  young_design = np.zeros(num_bunny_tetrahedra, dtype=np.float64)
  poisson_design = np.zeros(num_bunny_tetrahedra, dtype=np.float64)
  mass_design = np.zeros(num_bunny_vertices, dtype=np.float64)
  baseline_trajectory = run_forward(young_design, poisson_design, mass_design, adjoint_round=0)
  trajectory = baseline_trajectory
  initial_loss = trajectory["loss"]
  best_loss = initial_loss
  best_adjoint_round = 0
  best_young_design = young_design.copy()
  best_poisson_design = poisson_design.copy()
  best_mass_design = mass_design.copy()
  best_vertex_masses = trajectory["vertex_masses"].copy()
  best_terminal_position_summary = trajectory["terminal_position_summary"]
  best_floor_collision_summary = trajectory["floor_collision_summary"]
  young_adam_first_moment = np.zeros_like(young_design)
  young_adam_second_moment = np.zeros_like(young_design)
  poisson_adam_first_moment = np.zeros_like(poisson_design)
  poisson_adam_second_moment = np.zeros_like(poisson_design)
  mass_adam_first_moment = np.zeros_like(mass_design)
  mass_adam_second_moment = np.zeros_like(mass_design)
else:
  resumed_from_checkpoint = os.path.abspath(args.resume_checkpoint)
  if not os.path.isfile(resumed_from_checkpoint):
    raise FileNotFoundError(f"Resume checkpoint does not exist: {resumed_from_checkpoint}")
  required_checkpoint_values = {
    "completed_adjoint_round", "initial_loss", "current_loss",
    "young_design", "poisson_design", "mass_design",
    "young_adam_first_moment", "young_adam_second_moment",
    "poisson_adam_first_moment", "poisson_adam_second_moment",
    "mass_adam_first_moment", "mass_adam_second_moment",
    "best_loss", "best_adjoint_round", "best_young_design",
    "best_poisson_design", "best_mass_design",
  }
  with np.load(resumed_from_checkpoint, allow_pickle=False) as checkpoint:
    missing_checkpoint_values = required_checkpoint_values.difference(checkpoint.files)
    if missing_checkpoint_values:
      raise ValueError(f"Resume checkpoint is missing: {sorted(missing_checkpoint_values)}")
    resume_completed_adjoint_round = int(checkpoint["completed_adjoint_round"].item())
    initial_loss = float(checkpoint["initial_loss"].item())
    checkpoint_current_loss = float(checkpoint["current_loss"].item())
    young_design = checkpoint["young_design"].copy()
    poisson_design = checkpoint["poisson_design"].copy()
    mass_design = checkpoint["mass_design"].copy()
    young_adam_first_moment = checkpoint["young_adam_first_moment"].copy()
    young_adam_second_moment = checkpoint["young_adam_second_moment"].copy()
    poisson_adam_first_moment = checkpoint["poisson_adam_first_moment"].copy()
    poisson_adam_second_moment = checkpoint["poisson_adam_second_moment"].copy()
    mass_adam_first_moment = checkpoint["mass_adam_first_moment"].copy()
    mass_adam_second_moment = checkpoint["mass_adam_second_moment"].copy()
    best_loss = float(checkpoint["best_loss"].item())
    best_adjoint_round = int(checkpoint["best_adjoint_round"].item())
    best_young_design = checkpoint["best_young_design"].copy()
    best_poisson_design = checkpoint["best_poisson_design"].copy()
    best_mass_design = checkpoint["best_mass_design"].copy()
  expected_shapes = {
    "young_design": (young_design, (num_bunny_tetrahedra,)),
    "poisson_design": (poisson_design, (num_bunny_tetrahedra,)),
    "mass_design": (mass_design, (num_bunny_vertices,)),
    "young_adam_first_moment": (young_adam_first_moment, (num_bunny_tetrahedra,)),
    "young_adam_second_moment": (young_adam_second_moment, (num_bunny_tetrahedra,)),
    "poisson_adam_first_moment": (poisson_adam_first_moment, (num_bunny_tetrahedra,)),
    "poisson_adam_second_moment": (poisson_adam_second_moment, (num_bunny_tetrahedra,)),
    "mass_adam_first_moment": (mass_adam_first_moment, (num_bunny_vertices,)),
    "mass_adam_second_moment": (mass_adam_second_moment, (num_bunny_vertices,)),
  }
  for checkpoint_name, (checkpoint_value, expected_shape) in expected_shapes.items():
    if checkpoint_value.shape != expected_shape:
      raise ValueError(f"Checkpoint {checkpoint_name} has shape {checkpoint_value.shape}, expected {expected_shape}.")
  if resume_completed_adjoint_round < 0 or resume_completed_adjoint_round >= args.adjoint_steps:
    raise ValueError(f"Checkpoint completed round {resume_completed_adjoint_round} cannot resume to --adjoint-steps {args.adjoint_steps}.")
  print(f"Reconstructing converged trajectory for checkpoint round {resume_completed_adjoint_round:02d}.")
  trajectory = run_forward(young_design, poisson_design, mass_design)
  reconstructed_relative_difference = abs(trajectory["loss"] - checkpoint_current_loss) / max(1.0, abs(checkpoint_current_loss))
  resume_reconstructed_loss = trajectory["loss"]
  resume_reconstructed_loss_relative_difference = reconstructed_relative_difference
  print(f"Resumed checkpoint round {resume_completed_adjoint_round:02d}: " f"stored_loss={checkpoint_current_loss:.8e} reconstructed_loss={trajectory['loss']:.8e} " f"relative_difference={reconstructed_relative_difference:.8e}.")
  baseline_video_path = os.path.join(output_directory, "baseline.mp4")
  baseline_trajectory = {
    "max_newton_iterations": REFRESH_MAX_NEWTON_ITERATIONS,
    "gpu_memory": None,
    "frame_loss_summary": None,
    "motion_summary": None,
    "floor_collision_summary": None,
    "centroid_height_summary": None,
    "video": baseline_video_path if os.path.isfile(baseline_video_path) else None,
    "terminal_position_summary": None,
    "mass_summary": mass_statistics(initial_vertex_masses),
  }
  best_vertex_masses, _ = mass_values_from_design(best_mass_design, initial_vertex_masses)
  if best_adjoint_round == resume_completed_adjoint_round:
    best_terminal_position_summary = trajectory["terminal_position_summary"]
    best_floor_collision_summary = trajectory["floor_collision_summary"]
  else:
    best_terminal_position_summary = None
    best_floor_collision_summary = None

for adjoint_round in range(resume_completed_adjoint_round + 1, args.adjoint_steps + 1):
  (
    young_value_gradient,
    poisson_value_gradient,
    mass_value_gradient,
    backward_seconds,
    backward_timing,
    backward_memory,
  ) = run_backward(trajectory, adjoint_round)
  young_gradient = young_value_gradient * trajectory["young_chain"]
  poisson_gradient = (
    poisson_value_gradient * trajectory["poisson_chain"]
  )
  mass_gradient = mass_design_gradient(mass_value_gradient, trajectory["mass_chain"])

  if (
    not np.all(np.isfinite(young_gradient))
    or not np.all(np.isfinite(poisson_gradient))
    or not np.all(np.isfinite(mass_gradient))
    or maximum_element_norm(young_gradient) == 0.0
    or maximum_element_norm(poisson_gradient) == 0.0
    or np.max(np.abs(mass_gradient)) == 0.0
  ):
    print("Stopping because a joint design gradient is zero or non-finite.")
    break

  previous_loss = trajectory["loss"]
  previous_material_summary = trajectory["material_summary"]
  previous_mass_summary = trajectory["mass_summary"]
  young_adam_first_moment = ADAM_BETA1 * young_adam_first_moment + (1.0 - ADAM_BETA1) * young_gradient
  young_adam_second_moment = ADAM_BETA2 * young_adam_second_moment + (1.0 - ADAM_BETA2) * young_gradient * young_gradient
  poisson_adam_first_moment = ADAM_BETA1 * poisson_adam_first_moment + (1.0 - ADAM_BETA1) * poisson_gradient
  poisson_adam_second_moment = ADAM_BETA2 * poisson_adam_second_moment + (1.0 - ADAM_BETA2) * poisson_gradient * poisson_gradient
  mass_adam_first_moment = ADAM_BETA1 * mass_adam_first_moment + (1.0 - ADAM_BETA1) * mass_gradient
  mass_adam_second_moment = ADAM_BETA2 * mass_adam_second_moment + (1.0 - ADAM_BETA2) * mass_gradient * mass_gradient
  young_corrected_first = young_adam_first_moment / (1.0 - ADAM_BETA1 ** adjoint_round)
  young_corrected_second = young_adam_second_moment / (1.0 - ADAM_BETA2 ** adjoint_round)
  poisson_corrected_first = poisson_adam_first_moment / (1.0 - ADAM_BETA1 ** adjoint_round)
  poisson_corrected_second = poisson_adam_second_moment / (1.0 - ADAM_BETA2 ** adjoint_round)
  mass_corrected_first = mass_adam_first_moment / (1.0 - ADAM_BETA1 ** adjoint_round)
  mass_corrected_second = mass_adam_second_moment / (1.0 - ADAM_BETA2 ** adjoint_round)
  raw_young_update = -args.adam_learning_rate * young_corrected_first / (np.sqrt(young_corrected_second) + ADAM_EPSILON)
  raw_poisson_update = -args.adam_learning_rate * poisson_corrected_first / (np.sqrt(poisson_corrected_second) + ADAM_EPSILON)
  applied_mass_log_step = -args.mass_adam_learning_rate * mass_corrected_first / (np.sqrt(mass_corrected_second) + ADAM_EPSILON)
  proposed_young_step_norm = maximum_element_norm(raw_young_update)
  young_update_scale = min(1.0, args.young_step_size / proposed_young_step_norm)
  applied_young_step = young_update_scale * raw_young_update
  proposed_poisson_step_norm = maximum_element_norm(raw_poisson_update)
  poisson_update_scale = min(1.0, args.poisson_step_size / proposed_poisson_step_norm)
  applied_poisson_step = poisson_update_scale * raw_poisson_update

  young_design = young_design + applied_young_step
  young_design = np.clip(young_design, -YOUNG_LOGIT_DESIGN_LIMIT, YOUNG_LOGIT_DESIGN_LIMIT,)
  poisson_design = poisson_design + applied_poisson_step
  poisson_design = np.clip(poisson_design, -POISSON_LOGIT_DESIGN_LIMIT, POISSON_LOGIT_DESIGN_LIMIT,)
  mass_design = mass_design + applied_mass_log_step

  trajectory = run_forward(young_design, poisson_design, mass_design, adjoint_round=adjoint_round, baseline_loss=initial_loss,)
  loss_decreased = trajectory["loss"] < previous_loss

  record = {
    "adjoint_round": adjoint_round,
    "loss_before": previous_loss,
    "loss_after": trajectory["loss"],
    "relative_loss": trajectory["loss"] / initial_loss,
    "relative_reduction": 1.0 - trajectory["loss"] / initial_loss,
    "proposed_young_step_norm": proposed_young_step_norm,
    "young_step_was_capped": (
      proposed_young_step_norm > args.young_step_size
    ),
    "maximum_young_step_norm": maximum_element_norm(applied_young_step),
    "proposed_poisson_step_norm": proposed_poisson_step_norm,
    "poisson_step_was_capped": (
      proposed_poisson_step_norm > args.poisson_step_size
    ),
    "maximum_poisson_step_norm": maximum_element_norm(applied_poisson_step),
    "mass_log_step_inf": float(np.max(np.abs(applied_mass_log_step))),
    "loss_decreased": loss_decreased,
    "forward_seconds": trajectory["elapsed_seconds"],
    "forward_max_newton_iterations": trajectory["max_newton_iterations"],
    "forward_gpu_memory": trajectory["gpu_memory"],
    "frame_loss_summary": trajectory["frame_loss_summary"],
    "motion_summary": trajectory["motion_summary"],
    "floor_collision_summary": trajectory["floor_collision_summary"],
    "centroid_height_summary": trajectory["centroid_height_summary"],
    "terminal_position_summary": trajectory[
      "terminal_position_summary"
    ],
    "backward_seconds": backward_seconds,
    "backward_timing": backward_timing,
    "backward_gpu_memory": backward_memory,
    "adjoint_solver": "gpu_cg",
    "adjoint_solver_count": args.num_frames,
    "video": trajectory["video"],
  }
  record["gradient_statistics"] = {
    "young": {
      "transformed_l2": float(np.linalg.norm(young_gradient)),
      "transformed_inf": float(np.max(np.abs(young_gradient))),
      "value_l2": float(np.linalg.norm(young_value_gradient)),
      "value_inf": float(np.max(np.abs(young_value_gradient))),
    },
    "poisson": {
      "transformed_l2": float(np.linalg.norm(poisson_gradient)),
      "transformed_inf": float(np.max(np.abs(poisson_gradient))),
      "value_l2": float(np.linalg.norm(poisson_value_gradient)),
      "value_inf": float(np.max(np.abs(poisson_value_gradient))),
    },
    "mass": {
      "log_mass_l2": float(np.linalg.norm(mass_gradient)),
      "log_mass_inf": float(np.max(np.abs(mass_gradient))),
      "value_l2": float(np.linalg.norm(mass_value_gradient)),
      "value_inf": float(np.max(np.abs(mass_value_gradient))),
    },
  }
  record["material_statistics_before"] = previous_material_summary
  record["material_statistics_after"] = trajectory["material_summary"]
  record["mass_statistics_before"] = previous_mass_summary
  record["mass_statistics_after"] = trajectory["mass_summary"]
  optimization_history.append(record)
  is_new_best = trajectory["loss"] < best_loss
  if is_new_best:
    best_loss = trajectory["loss"]
    best_adjoint_round = adjoint_round
    best_young_design = young_design.copy()
    best_poisson_design = poisson_design.copy()
    best_mass_design = mass_design.copy()
    best_vertex_masses = trajectory["vertex_masses"].copy()
    best_terminal_position_summary = trajectory["terminal_position_summary"]
    best_floor_collision_summary = trajectory["floor_collision_summary"]
  if args.save_obj:
    candidate_directory = os.path.join(output_directory, "_candidate_best")
    if is_new_best:
      best_obj_directory = os.path.join(output_directory, "best")
      if os.path.isdir(best_obj_directory):
        shutil.rmtree(best_obj_directory)
      os.replace(candidate_directory, best_obj_directory)
    elif os.path.isdir(candidate_directory):
      shutil.rmtree(candidate_directory)
  checkpoint_path = os.path.join(output_directory, "latest_checkpoint.npz")
  checkpoint_temporary_path = os.path.join(output_directory, "latest_checkpoint.tmp.npz")
  checkpoint_payload = {
    "completed_adjoint_round": adjoint_round,
    "initial_loss": initial_loss,
    "current_loss": trajectory["loss"],
    "young_design": young_design,
    "poisson_design": poisson_design,
    "mass_design": mass_design,
    "young_adam_first_moment": young_adam_first_moment,
    "young_adam_second_moment": young_adam_second_moment,
    "poisson_adam_first_moment": poisson_adam_first_moment,
    "poisson_adam_second_moment": poisson_adam_second_moment,
    "mass_adam_first_moment": mass_adam_first_moment,
    "mass_adam_second_moment": mass_adam_second_moment,
    "best_loss": best_loss,
    "best_adjoint_round": best_adjoint_round,
    "best_young_design": best_young_design,
    "best_poisson_design": best_poisson_design,
    "best_mass_design": best_mass_design,
  }
  np.savez(checkpoint_temporary_path, **checkpoint_payload)
  os.replace(checkpoint_temporary_path, checkpoint_path)
  print(f"Saved optimization checkpoint: {checkpoint_path}")
  optimization_message = (
    f"optimization={adjoint_round:02d}/{args.adjoint_steps:02d} "
    f"loss={previous_loss:.8e}->{trajectory['loss']:.8e} "
    f"relative_loss={trajectory['loss'] / initial_loss:.8e} "
    f"relative_reduction={1.0 - trajectory['loss'] / initial_loss:.8e} "
    f"{format_material_statistics(trajectory['material_summary'])} "
    f"{format_mass_statistics(trajectory['mass_summary'])} "
    f"young_step={maximum_element_norm(applied_young_step):.6e} "
    f"poisson_step={maximum_element_norm(applied_poisson_step):.6e} "
    f"mass_log_step={float(np.max(np.abs(applied_mass_log_step))):.6e}"
  )
  print(optimization_message)
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
  "completed_adjoint_steps": resume_completed_adjoint_round + len(optimization_history),
  "resumed_from_checkpoint": resumed_from_checkpoint,
  "resume_completed_adjoint_round": resume_completed_adjoint_round,
  "resume_reconstructed_loss": resume_reconstructed_loss,
  "resume_reconstructed_loss_relative_difference": resume_reconstructed_loss_relative_difference,
  "history_scope": "resumed segment only" if resumed_from_checkpoint is not None else "complete run",
  "optimization_target": OPTIMIZATION_TARGET,
  "adjoint_type": "approximate_projected_hessian",
  "adjoint_solver": "gpu_cg",
  "optimizer": "adam",
  "young_step_size": args.young_step_size,
  "poisson_step_size": args.poisson_step_size,
  "initial_total_mass": BUNNY_TOTAL_MASS,
  "mass_total_constraint": BUNNY_TOTAL_MASS,
  "mass_parameterization": "per-vertex 10%-of-initial floor plus normalized positive free mass",
  "initial_mass_distribution": "rest-tetrahedron volume lumping, one quarter of each tetrahedron mass per incident vertex",
  "bunny_rest_total_volume": bunny_total_volume,
  "bunny_density": bunny_density,
  "initial_vertex_mass_statistics": mass_statistics(initial_vertex_masses),
  "minimum_vertex_mass": float(minimum_vertex_masses.min()),
  "adam_learning_rate": args.adam_learning_rate,
  "mass_adam_learning_rate": args.mass_adam_learning_rate,
  "adam_beta1": ADAM_BETA1,
  "adam_beta2": ADAM_BETA2,
  "adam_epsilon": ADAM_EPSILON,
  "loss_definition": (
    "the weighted target-position loss at "
    + ("only the final converged frame" if args.target_loss_final_frame_only else "every converged frame")
    + ", plus the squared desired 3D displacement-magnitude loss at every converged frame"
  ),
  "target_loss_schedule": "final-frame-only" if args.target_loss_final_frame_only else "every-frame",
  "floor_center": FLOOR_CENTER.tolist(),
  "floor_collision_diagnostic": "strict per-frame count of bunny vertices with y below floor height; diagnostic only, not an energy",
  "loss_target": LOSS_TARGET.tolist(),
  "loss_weight": loss_weight_value,
  "motion_loss_weight": MOTION_LOSS_WEIGHT_VALUE,
  "motion_loss_axis": "xyz-magnitude",
  "desired_displacement_per_frame": DESIRED_DISPLACEMENT_VALUE,
  "desired_speed": DESIRED_DISPLACEMENT_VALUE / DT_VALUE,
  "motion_norm_epsilon": MOTION_NORM_EPSILON,
  "initial_loss": initial_loss,
  "baseline_max_newton_iterations": baseline_trajectory["max_newton_iterations"],
  "baseline_gpu_memory": baseline_trajectory["gpu_memory"],
  "baseline_frame_loss_summary": baseline_trajectory[
    "frame_loss_summary"
  ],
  "baseline_motion_summary": baseline_trajectory["motion_summary"],
  "baseline_floor_collision_summary": baseline_trajectory[
    "floor_collision_summary"
  ],
  "baseline_centroid_height_summary": baseline_trajectory[
    "centroid_height_summary"
  ],
  "baseline_video": baseline_trajectory["video"],
  "baseline_terminal_position_summary": baseline_trajectory[
    "terminal_position_summary"
  ],
  "baseline_mass_statistics": baseline_trajectory["mass_summary"],
  "best_loss": best_loss,
  "best_adjoint_round": best_adjoint_round,
  "best_terminal_position_summary": best_terminal_position_summary,
  "best_floor_collision_summary": best_floor_collision_summary,
  "final_loss": trajectory["loss"],
  "final_terminal_position_summary": trajectory[
    "terminal_position_summary"
  ],
  "final_floor_collision_summary": trajectory[
    "floor_collision_summary"
  ],
  "latest_checkpoint_file": os.path.join(output_directory, "latest_checkpoint.npz",),
  "history": optimization_history,
  "total_seconds": time.time() - optimization_start,
}
best_young_values, _ = young_values_from_design(best_young_design)
best_poisson_values, _ = poisson_values_from_design(best_poisson_design)
best_material_values = material_values_from_parameters(best_young_values, best_poisson_values,)
best_young_path = os.path.join(output_directory, "best_young.npy")
final_young_path = os.path.join(output_directory, "final_young.npy")
best_young_design_path = os.path.join(output_directory, "best_young_design.npy")
final_young_design_path = os.path.join(output_directory, "final_young_design.npy")
best_poisson_path = os.path.join(output_directory, "best_poisson.npy")
final_poisson_path = os.path.join(output_directory, "final_poisson.npy")
best_design_path = os.path.join(output_directory, "best_poisson_design.npy")
final_design_path = os.path.join(output_directory, "final_poisson_design.npy")
best_mass_path = os.path.join(output_directory, "best_vertex_mass.npy")
final_mass_path = os.path.join(output_directory, "final_vertex_mass.npy")
best_mass_design_path = os.path.join(output_directory, "best_mass_design.npy")
final_mass_design_path = os.path.join(output_directory, "final_mass_design.npy")
np.save(best_young_path, best_young_values)
np.save(final_young_path, trajectory["young_values"])
np.save(best_young_design_path, best_young_design)
np.save(final_young_design_path, young_design)
np.save(best_poisson_path, best_poisson_values)
np.save(final_poisson_path, trajectory["poisson_values"])
np.save(best_design_path, best_poisson_design)
np.save(final_design_path, poisson_design)
np.save(best_mass_path, best_vertex_masses)
np.save(final_mass_path, trajectory["vertex_masses"])
np.save(best_mass_design_path, best_mass_design)
np.save(final_mass_design_path, mass_design)
results["young_parameterization"] = {
  "differentiated_value": "young",
  "layout": "one Young's modulus per tetrahedron",
  "num_tetrahedra": num_bunny_tetrahedra,
  "design_value": "bounded_young_logit_offset",
  "young_bounds": [YOUNG_MIN_VALUE, YOUNG_MAX_VALUE],
}
results["poisson_parameterization"] = {
  "differentiated_value": "poisson",
  "layout": "one Poisson ratio per tetrahedron",
  "num_tetrahedra": num_bunny_tetrahedra,
  "design_value": "bounded_poisson_logit_offset",
  "poisson_bounds": [POISSON_MIN_VALUE, POISSON_MAX_VALUE],
}
results["mass_parameterization"] = {
  "differentiated_value": "mass",
  "layout": "one mass per bunny vertex",
  "num_vertices": num_bunny_vertices,
  "design_value": "log((mass-minimum_mass)/(initial_mass-minimum_mass))",
  "minimum_definition": "each vertex retains at least 10% of its initial mass",
  "minimum_statistics": mass_statistics(minimum_vertex_masses),
  "constraint": f"mass is at least its fixed per-vertex minimum and total mass remains {BUNNY_TOTAL_MASS}",
}
results["young_jacobian"] = {
  "rows": young_jacobian.rows,
  "cols": young_jacobian.cols,
  "static_block_dimensions": young_jacobian.block_dimensions,
  "static_block_counts": young_jacobian.block_counts,
}
results["poisson_jacobian"] = {
  "rows": poisson_jacobian.rows,
  "cols": poisson_jacobian.cols,
  "static_block_dimensions": poisson_jacobian.block_dimensions,
  "static_block_counts": poisson_jacobian.block_counts,
}
results["mass_jacobian"] = {
  "rows": mass_jacobian.rows,
  "cols": mass_jacobian.cols,
  "static_block_dimensions": mass_jacobian.block_dimensions,
  "static_block_counts": mass_jacobian.block_counts,
}
results["base_material_statistics"] = material_statistics(base_material_values)
results["best_material_statistics"] = material_statistics(best_material_values)
results["final_material_statistics"] = trajectory["material_summary"]
results["best_mass_statistics"] = mass_statistics(best_vertex_masses)
results["final_mass_statistics"] = trajectory["mass_summary"]
results["best_young_values_file"] = best_young_path
results["final_young_values_file"] = final_young_path
results["best_young_design_file"] = best_young_design_path
results["final_young_design_file"] = final_young_design_path
results["best_poisson_values_file"] = best_poisson_path
results["final_poisson_values_file"] = final_poisson_path
results["best_poisson_design_file"] = best_design_path
results["final_poisson_design_file"] = final_design_path
results["best_vertex_mass_file"] = best_mass_path
results["final_vertex_mass_file"] = final_mass_path
results["best_mass_design_file"] = best_mass_design_path
results["final_mass_design_file"] = final_mass_design_path
baseline_surface_obj_directory = os.path.join(output_directory, "baseline", "bunny_obj")
best_surface_obj_directory = baseline_surface_obj_directory if best_adjoint_round == 0 else os.path.join(output_directory, "best", "bunny_obj")
results["surface_obj_exports"] = {
  "enabled": args.save_obj,
  "surface_only": True,
  "num_surface_vertices": int(bunny_surface_indices.size),
  "num_surface_triangles": int(bunny_surface_triangles_local.shape[0]),
  "baseline_all_frames_directory": baseline_surface_obj_directory if args.save_obj else None,
  "best_all_frames_directory": best_surface_obj_directory if args.save_obj else None,
  "best_adjoint_round": best_adjoint_round,
}
results_path = os.path.join(output_directory, "results.json")
with open(results_path, "w", encoding="utf-8") as file:
  json.dump(results, file, indent=2)

print(f"Inverse simulation finished in {results['total_seconds']:.2f}s: " f"loss={initial_loss:.8e}->{trajectory['loss']:.8e}, " f"target={OPTIMIZATION_TARGET}.")
print(f"Saved optimization results: {results_path}")
if not args.no_gui:
  plotter.close()
