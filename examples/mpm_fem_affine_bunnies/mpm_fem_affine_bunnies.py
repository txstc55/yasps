"""Coupled implicit MPM, FEM, and affine bunnies. Run this file directly."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from yasps import attribute, differentiator, matrix, scene, vector
from yasps.solver import solver
import yasps.helper
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import pyvista as pv

from helpers import GRID_N, GRID, DX, GRID_ORIGIN, load_bunny, place_bunny, vertex_masses, volume_samples, check_aabbs, surface_triangles, surface_edges, container_mesh
from helpers import BOX_LOWER, BOX_UPPER, TRANSFER_OPTIONS
from helpers import constant, inertia, edge_matrix, snh_density, snh_parameters, rotation_penalty, particle_graph, freeze_particle_model, bind_particle_range, FrameHierarchyTopology, assemble_batched, full_energy
from helpers import contact_barrier, contact_distance_squared, update_contacts, save_video
from helpers import position_attributes, update_position, create_collision_detector, ccd_sweep_with_growth, liquid_volume_from_deformation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ccd"))

##################################################################
## Configuration: SI, y-up, 64 nodes per axis, nine bodies in total
##################################################################
FRAME_DT = 0.01
NUM_FRAMES = 400
PARTICLES_PER_BUNNY = 80_000
LIQUID_PARTICLES_PER_BUNNY = [231_525, 200_000, 150_000]
MOTION_TOLERANCE = 2e-2
CG_TOLERANCE = 1e-7
MAX_CG_ITERATIONS = 2000
MAX_NEWTON_ITERATIONS = 10000
GRID_INERTIA_MIN_MASS = 1e-8
MAX_BACKTRACKS = 8
MATERIAL_BATCH_SIZE = 100_000  # Bound raw-coordinate sorting memory; liquid uses six batches.
SEPARATE_COORDINATES = False
HIERARCHY_INCLUDE_INITIAL_CONTACTS = True
HIERARCHY_REBUILD_INTERVAL = 1
CONTACT_DISTANCE = 0.001  # physical activation distance; CCD receives its square
CONTACT_STIFFNESS = 1e7
CCD_SLACKNESS = 0.5
CONTAINER_MESH_ID = 5  # Distinct from both affine bodies (2/3) and MPM (4).
CHECKPOINT_START_FRAME = 10
ORTHOGONAL_STIFFNESS = 1e9
DETERMINANT_STIFFNESS = 1e9
INITIAL_VELOCITY = np.zeros(3)
SIZES = [.10, .13, .16, .15, .21, .20, .16, .19, .20 * (150_000 / 200_000)**(1 / 3)]
# Shift nearby bodies to clear the enlarged third liquid bunny at the front.
CENTERS = [[-.13, .38, -.13], [.13, .14, -.13], [-.1816012, .38, .13], [.13, .14, .13], [-.1272443, .14, -.13], [.13, .38, -.13], [-.13, .1015031, .13], [.1336753, .38, .13], [-.035024, .27, .17]]
ANGLES = [[10, 30, 5], [-15, -20, 20], [5, 80, -10], [20, 140, 5], [-10, 10, 15], [15, -40, -5], [-15, 60, 10], [5, -70, -15], [0, 25, 0]]
YOUNG = [2e4, 5e4]
POISSON = [.45, .25]
LIQUID_BULK = 2e3
LIQUID_DENSITY = 1000.0
# Ratios use ACTUAL body masses, not just densities: sizes/volumes differ.
AFFINE_MASS_MULTIPLIER = 2.0  # times the heavier liquid bunny
SOFT_MASS_MULTIPLIER = 0.4  # times the lighter liquid bunny
SOLID_MPM_MASS_MULTIPLIER = 0.05  # times the lighter liquid bunny
COLORS = ["#dfad3b", "#cb792d", "#e64b4b", "#44aa66", "#38bde0", "#38bde0", "#7855cf", "#ef77ad", "#38bde0"]
MPM_BODY_GROUPS = [("liquid", [4, 5, 8]), ("solid", [6, 7])]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--frames", type=int, default=NUM_FRAMES)
parser.add_argument("--particles", type=int, default=PARTICLES_PER_BUNNY, help="Particles per solid MPM bunny")
parser.add_argument("--liquid-particles", type=int, nargs=3, default=LIQUID_PARTICLES_PER_BUNNY, help="Particle counts for liquid0, liquid1, liquid2")
parser.add_argument("--batch-size", type=int, default=MATERIAL_BATCH_SIZE, help="Particles per bounded MPM material-Hessian batch")
parser.add_argument("--output", type=Path, default=HERE / "outputs" / "coupled")
parser.add_argument("--restart", type=Path, help="Resume from a complete restart_after_frame_XXXX.npz checkpoint")
parser.add_argument("--gui", action="store_true")
parser.add_argument("--checkpoints-only", action="store_true", help="Save restart NPZs including liquid_J from frame 0; no rendering or other exports")
parser.add_argument("--fresh-cache", action="store_true")
args = parser.parse_args()
yasps.helper.DEBUG_TIME = False  # The frame log below records phase timings without per-kernel print overhead.
if min(args.frames, args.particles, *args.liquid_particles, args.batch_size, HIERARCHY_REBUILD_INTERVAL) <= 0:
  raise ValueError("Frames, particle count, and batch size must be positive")
particles_per_material = {"liquid": args.liquid_particles, "solid": [args.particles, args.particles]}
total_particles = sum(sum(counts) for counts in particles_per_material.values())
OUTPUT = args.output.resolve()
OUTPUT.mkdir(parents=True, exist_ok=True)
os.chdir(HERE)
if args.fresh_cache:
  backup = Path(tempfile.mkdtemp(prefix="yasps_mpm_cache_backup_"))
  for name in (".yasps_tmp", ".yasps_constant"):
    if (HERE / name).exists():
      (HERE / name).rename(backup / name)
  print(f"Old generated cache moved to {backup}", flush=True)
started = time.perf_counter()
free_at_start, gpu_total = cuda.mem_get_info()
peak_used = gpu_total - free_at_start
transfer = SourceModule((HERE / "transfers.cu").read_text(), options=TRANSFER_OPTIONS, no_extern_c=True)
rng = np.random.default_rng(1313)

##################################################################
## Load positive tets, place nine bunnies, and distribute physical mass
##################################################################
source_vertices, source_tets = load_bunny(HERE.parent / "data")
triangles = surface_triangles(source_tets)
surface_ids = np.unique(triangles)
surface_remap = np.full(len(source_vertices), -1, dtype=np.int64)
surface_remap[surface_ids] = np.arange(len(surface_ids))
render_triangles = surface_remap[triangles]
rest, positions, transforms, volumes = [], [], [], []
for i in range(len(SIZES)):
  X, x, T = place_bunny(source_vertices, source_tets, SIZES[i], CENTERS[i], ANGLES[i])
  rest.append(X)
  positions.append(x)
  transforms.append(T)
  volumes.append(vertex_masses(X, source_tets, 1.0)[1])
check_aabbs(positions)
for points in positions[4:]:
  stencil_base = np.floor((points - GRID_ORIGIN) / DX - 0.5).astype(np.int64)
  if np.any(stencil_base < 0) or np.any(stencil_base + 2 >= GRID_N):
    raise ValueError("An initial MPM bunny has a 27-node stencil outside the grid")
liquid_masses = np.array([volumes[i].sum() * LIQUID_DENSITY for i in (4, 5)])
body_masses = np.r_[np.full(2, AFFINE_MASS_MULTIPLIER * liquid_masses.max()), np.full(2, SOFT_MASS_MULTIPLIER * liquid_masses.min()), liquid_masses, np.full(2, SOLID_MPM_MASS_MULTIPLIER * liquid_masses.min()), volumes[8].sum() * LIQUID_DENSITY]
densities = body_masses / np.array([v.sum() for v in volumes])
nv = len(source_vertices)
specs = {"frames": args.frames, "nominal_dt": FRAME_DT, "grid_nodes": GRID_N, "grid_spacing": DX, "particles_per_bunny": args.particles, "total_particles": 4 * args.particles, "material_batch_size": args.batch_size, "body_order": ["affine0", "affine1", "soft0", "soft1", "liquid0", "liquid1", "solid_mpm0", "solid_mpm1"], "masses_kg": body_masses.tolist(), "densities_kg_m3": densities.tolist(), "body_volumes_m3": [float(v.sum()) for v in volumes], "colors": COLORS, "snh_young_pa": YOUNG, "poisson": POISSON, "liquid_bulk_pa": LIQUID_BULK, "contact_distance_m": CONTACT_DISTANCE, "contact_stiffness": CONTACT_STIFFNESS, "smoke_test": args.particles != PARTICLES_PER_BUNNY}
specs.update({"linear_solver": "mas", "fallback_solver": "jacobian", "grid_inertia_min_mass": GRID_INERTIA_MIN_MASS, "cg_tolerance": CG_TOLERANCE, "max_cg_iterations": MAX_CG_ITERATIONS, "max_newton_iterations": MAX_NEWTON_ITERATIONS, "max_backtracks": MAX_BACKTRACKS, "ccd_slackness": CCD_SLACKNESS, "fixed_timestep": True})
specs.update({"determinant_step_limit": False, "material_determinant_barrier": False, "determinant_barrier_materials": [], "grouped_add_particle_material": True, "grouped_add_affine_vertex_inertia": True, "grouped_add_collision": False})
specs.update({"liquid_determinant_penalty": False, "total_particles": total_particles, "particles_per_bunny": particles_per_material, "smoke_test": args.particles != PARTICLES_PER_BUNNY or args.liquid_particles != LIQUID_PARTICLES_PER_BUNNY, "body_order": ["affine0", "affine1", "soft0", "soft1", "liquid0", "liquid1", "solid_mpm0", "solid_mpm1", "liquid2"], "ccd_capacity_growth": 1.5})
specs.update({"position_attribute_layout": "three scalar attributes per vertex" if SEPARATE_COORDINATES else "one 1x3 attribute per vertex", "affine_unknown_layout": "three 1x4 rows", "mpm_solid_model": "projected stable Neo-Hookean", "mpm_liquid_model": "frozen scalar volume/divergence GN residual", "liquid_volume_state": "J_next=J_previous*det(I+D), legacy checkpoint migration uses 80-digit determinant", "liquid_inner_hessian": "1x1 residual, projection_method=-1, 1x1 product tiles"})
specs.update({"mas_hierarchy_topology": "static scene, frozen particle and initial contact Hessian coordinates" if HIERARCHY_INCLUDE_INITIAL_CONTACTS else "static scene and frozen particle Hessian coordinates", "mas_hierarchy_rebuild": f"every {HIERARCHY_REBUILD_INTERVAL} frames after P2G", "mas_hierarchy_rebuild_interval": HIERARCHY_REBUILD_INTERVAL, "mas_hierarchy_includes_collision_graph": HIERARCHY_INCLUDE_INITIAL_CONTACTS})
specs.update({"container_collision": "fixed open-top triangle mesh in ordinary CCD/contact", "container_mesh_id": CONTAINER_MESH_ID, "container_bounds_m": [BOX_LOWER.tolist(), BOX_UPPER.tolist()], "collision_queries": "FEM and affine surface vertices; every MPM particle; container vertices", "analytic_wall_energy": False, "analytic_wall_step_limit": False, "checkpoint_start_frame": CHECKPOINT_START_FRAME})
specs.update({"grid_bounds_m": [GRID[0].tolist(), GRID[-1].tolist()], "grid_margin_m": DX / 2, "initial_velocity_m_s": INITIAL_VELOCITY.tolist(), "initial_centers_m": CENTERS, "affine_orthogonal_stiffness": ORTHOGONAL_STIFFNESS, "affine_determinant_stiffness": DETERMINANT_STIFFNESS})
specs.update({"inner_hessian_tiles": "all contact and MPM material energies", "inertia_assembly": "existing component method"})
if not args.checkpoints_only:
  (OUTPUT / "configuration.json").write_text(json.dumps(specs, indent=2))
print("CONFIGURATION " + json.dumps(specs), flush=True)

##################################################################
## Fixed grid vector unknowns; inertia mass is floored on the GPU each frame
##################################################################
s = scene("mpm_coupled")
mesh = s.addMesh("bodies")
h_attribute = constant(s, "h", [FRAME_DT])
dhat = constant(s, "dhat", [CONTACT_DISTANCE**2])
kappa = constant(s, "kappa", [CONTACT_STIFFNESS])
grid = mesh.addPrimitive("grid", numInstances=len(GRID))
grid_mass = constant(grid, "mass", np.zeros(len(GRID)))
grid_coordinates, q = position_attributes(grid, GRID, SEPARATE_COORDINATES)
G = constant(grid, "rest", GRID, 1, 3)
grid_reference = [constant(grid, f"rest_{name}", GRID[:, axis]) for axis, name in enumerate("xyz")] if SEPARATE_COORDINATES else G
qhat = constant(grid, "target", GRID, 1, 3)
grid_energy = grid.addAttribute("inertia", computed_attribute=inertia(q, qhat, grid_mass))
s.addEnergy(grid_energy, projection_method=-1)

##################################################################
## Soft FEM: per-vertex inertia and full 9-by-9 material projection
##################################################################
soft = mesh.addPrimitive("soft", numInstances=2 * nv)
soft_coordinates, soft_x = position_attributes(soft, np.concatenate(positions[2:4]), SEPARATE_COORDINATES)
soft_target = constant(soft, "target", np.concatenate(positions[2:4]), 1, 3)
soft_mass_cpu = np.concatenate([vertex_masses(rest[i], source_tets, densities[i])[0] for i in (2, 3)])
soft_mass = constant(soft, "mass", soft_mass_cpu)
soft_energy = soft.addAttribute("inertia", computed_attribute=inertia(soft_x, soft_target, soft_mass))
s.addEnergy(soft_energy, projection_method=-1)
soft_tets = mesh.addPrimitive("soft_tets", numInstances=2 * len(source_tets))
soft_indices = np.concatenate([source_tets, source_tets + nv])
soft_connection = soft_tets.addConnectivity("vertices", soft, soft_indices, 4)
soft_join = soft_tets.addAttribute("positions", through=soft_connection, source=soft_x)
soft_reference = np.concatenate(positions[2:4])[soft_indices]
Dm = (soft_reference[:, 1:] - soft_reference[:, :1]).transpose(0, 2, 1)
inverse_Dm = constant(soft_tets, "inverse_Dm", np.linalg.inv(Dm), 3, 3)
soft_F = soft_tets.addAttribute("F", computed_attribute=edge_matrix(soft_join) * inverse_Dm)
soft_material = mesh.addPrimitive("soft_material", numInstances=2 * len(source_tets))
soft_identity = soft_material.addConnectivity("tet", soft_tets, np.arange(2 * len(source_tets), dtype=np.uint32), 1)
soft_local_F = soft_material.addAttribute("F", through=soft_identity, source=soft_F).resize(3, 3)
soft_volume = constant(soft_material, "volume", np.concatenate(volumes[2:4]))
mu_cpu, lam_cpu = zip(*[snh_parameters(e, nu) for e, nu in zip(YOUNG, POISSON)])
soft_mu = constant(soft_material, "mu", np.repeat(mu_cpu, len(source_tets)))
soft_lam = constant(soft_material, "lam", np.repeat(lam_cpu, len(source_tets)))
soft_density = snh_density(soft_local_F, soft_mu, soft_lam)
soft_elastic = soft_material.addAttribute("elastic", computed_attribute=h_attribute * h_attribute * soft_volume * soft_density)
s.addEnergy(soft_elastic, projection_method=1, separate_hessian_jacobian=True)

##################################################################
## Affine bodies: three independent rows, volume-lumped vertex inertia
##################################################################
affine = mesh.addPrimitive("affine", numInstances=2)
affine_rows = []
for a, name in enumerate("xyz"):
  affine_rows.append(affine.addAttribute(f"T_{name}", rows=1, cols=4))
  affine_rows[-1].updateValue(np.array(transforms[:2])[:, a, :].copy().ravel())
T = affine.addAttribute("T", computed_attribute=attribute.to_array([row[b] for row in affine_rows for b in range(4)], rows=3, cols=4))
affine_volume = constant(affine, "volume", [volumes[i].sum() for i in (0, 1)])
affine_energy = affine.addAttribute("rotation", computed_attribute=h_attribute * h_attribute * affine_volume * rotation_penalty(T, ORTHOGONAL_STIFFNESS, DETERMINANT_STIFFNESS))
s.addEnergy(affine_energy, projection_method=1)
affine_vertices = mesh.addPrimitive("affine_vertices", numInstances=2 * nv)
affine_connection = affine_vertices.addConnectivity("body", affine, np.repeat(np.arange(2, dtype=np.uint32), nv), 1)
affine_rest = constant(affine_vertices, "rest", np.column_stack([np.concatenate(rest[:2]), np.ones(2 * nv)]), 4)
affine_coordinates = []
for a, name in enumerate("xyz"):
  row = affine_vertices.addAttribute(f"body_T_{name}", through=affine_connection, source=affine_rows[a]).resize(1, 4)
  affine_coordinates.append(affine_vertices.addAttribute(name, computed_attribute=(row * affine_rest)[0]))
affine_x = affine_vertices.addAttribute("position", computed_attribute=attribute.to_array(affine_coordinates, rows=1, cols=3))
affine_mass_cpu = np.concatenate([vertex_masses(rest[i], source_tets, densities[i])[0] for i in (0, 1)])
# A one-to-one join keeps this inertia's inner matrix 3x3, not 12x12.
affine_inertia = mesh.addPrimitive("affine_inertia", numInstances=2 * nv)
affine_identity = affine_inertia.addConnectivity("vertex", affine_vertices, np.arange(2 * nv, dtype=np.uint32), 1)
affine_inertia_x = affine_inertia.addAttribute("position", through=affine_identity, source=affine_x).resize(1, 3)
affine_target = constant(affine_inertia, "target", np.concatenate(positions[:2]), 1, 3)
affine_mass = constant(affine_inertia, "mass", affine_mass_cpu)
affine_kinetic = affine_inertia.addAttribute("inertia", computed_attribute=inertia(affine_inertia_x, affine_target, affine_mass))
s.addEnergy(affine_kinetic, projection_method=-1, separate_hessian_jacobian=True, grouped_add=True)

##################################################################
## Five MPM bunnies, one shared grid; liquid also stores volume ratio J
##################################################################
groups = []
for material_type, body_ids in MPM_BODY_GROUPS:
  particle_counts = particles_per_material[material_type]
  samples, particle_volumes = [], []
  for i, count in zip(body_ids, particle_counts):
    points, pv0 = volume_samples(positions[i], source_tets, count, rng)
    samples.append(points)
    particle_volumes.append(pv0)
  count = sum(particle_counts)
  group = particle_graph(mesh, material_type, grid, grid_coordinates if SEPARATE_COORDINATES else q, grid_reference, count, h_attribute, material_type == "liquid")
  group["body_ids"] = body_ids
  group["particle_counts"] = particle_counts
  group["body_offsets"] = np.r_[0, np.cumsum(particle_counts)]
  group["state"] = {"x": gpuarray.to_gpu(np.concatenate(samples).ravel()), "v": gpuarray.to_gpu(np.tile(INITIAL_VELOCITY, count)), "F": gpuarray.to_gpu(np.tile(np.eye(3).ravel(), count)), "C": gpuarray.zeros(count * 9, np.float64)}
  if group["liquid"]:
    group["state"]["J"] = gpuarray.to_gpu(np.ones(count))
  group["parameters"] = {"volume": gpuarray.to_gpu(np.concatenate(particle_volumes))}
  if material_type == "liquid":
    group["parameters"]["bulk"] = gpuarray.to_gpu(np.full(count, LIQUID_BULK))
  else:
    group["parameters"].update(mu=gpuarray.to_gpu(np.repeat(mu_cpu, particle_counts)), lam=gpuarray.to_gpu(np.repeat(lam_cpu, particle_counts)))
  group["mass"] = gpuarray.to_gpu(np.concatenate([particle_volumes[j] * densities[i] for j, i in enumerate(body_ids)]))
  group["frozen"] = dict(group["parameters"], x=group["state"]["x"], F=group["state"]["F"], indices=gpuarray.zeros(count * 27, np.uint32), weights=gpuarray.zeros(count * 27, np.float64), B=gpuarray.zeros(count * 81, np.float64))
  freeze_particle_model(group)
  groups.append(group)
# Unequal material counts must agree in state, mass, and every bound parameter.
for group in groups:
  assert len(group["particle_counts"]) == len(group["body_ids"])
  assert group["count"] == group["body_offsets"][-1] == group["mass"].size
  assert group["state"]["x"].size == 3 * group["count"] and all(value.size == group["count"] for value in group["parameters"].values())
targets = grid_coordinates + soft_coordinates + affine_rows

##################################################################
## One collision union, including the fixed container's actual triangles
##################################################################
container_points, container_triangles = container_mesh()
container = mesh.addPrimitive("container", numInstances=len(container_points))
if SEPARATE_COORDINATES:
  for axis, name in enumerate("xyz"):
    constant(container, name, container_points[:, axis])
constant(container, "position", container_points, 1, 3)
# Appending the fixed vertices preserves every existing body/particle offset.
union = mesh.addPrimitiveUnion("collision_vertices", [soft, affine_vertices, groups[0]["primitive"], groups[1]["primitive"], container])
if SEPARATE_COORDINATES:
  union_coordinates = [union.addAttribute(axis) for axis in "xyz"]
  union_x = union.addAttribute("position", computed_attribute=attribute.to_array(union_coordinates, rows=1, cols=3))
else:
  union_x = union.addAttribute("position")
offsets = np.cumsum([0, 2 * nv, 2 * nv, groups[0]["count"], groups[1]["count"], len(container_points)])
faces = np.concatenate([triangles, triangles + nv, triangles + 2 * nv, triangles + 3 * nv, container_triangles + offsets[4]]).astype(np.uint32)
body_query_ids = np.r_[surface_ids, surface_ids + nv, surface_ids + 2 * nv, surface_ids + 3 * nv, np.arange(offsets[2], offsets[4])].astype(np.uint32)
query_ids = np.r_[body_query_ids, np.arange(offsets[4], offsets[5])].astype(np.uint32)
edges = surface_edges(faces)
mesh_ids = np.r_[np.zeros(2 * nv), np.full(nv, 2), np.full(nv, 3), np.full(total_particles, 4), np.full(len(container_points), CONTAINER_MESH_ID)].astype(np.uint32)
assert mesh_ids.size == offsets[-1]
vertex_weight = gpuarray.to_gpu(np.r_[np.full(4 * nv, -1.0), groups[0]["parameters"]["volume"].get() / DX**3, groups[1]["parameters"]["volume"].get() / DX**3, np.full(len(container_points), -1.0)])
contacts = []
for name, arity in [("pp", 2), ("pe", 3), ("pt", 4), ("ee", 4)]:
  primitive = mesh.addPrimitive(name, numInstances=0, isDynamic=True)
  connection = primitive.addConnectivity("vertices", union, np.empty((0, arity), dtype=np.uint32), arity)
  local_position = primitive.addAttribute("positions", through=connection, source=union_x)
  weight = constant(primitive, "weight", np.empty(0))
  energy = primitive.addAttribute("barrier", computed_attribute=contact_barrier(local_position, name, dhat, h_attribute * h_attribute * kappa * weight))
  squared_distance = primitive.addAttribute("distance_squared", computed_attribute=contact_distance_squared(local_position, name))
  s.addEnergy(energy, projection_method=1, dynamic_instances=True, separate_hessian_jacobian=True, auto_partition=False)
  contacts.append({"primitive": primitive, "connection": connection, "weight": weight, "arity": arity, "distance": squared_distance})
# Register targets only after every scene energy, including contact, is added.
s.addMinimizeTarget(targets)
s.minimizer.setSolver("mas")
for group in groups:
  model = "liquid scalar GN residual" if group["liquid"] else "projected stable Neo-Hookean"
  print(f"DIFFERENTIATE {group['primitive'].name}: {model}, batch size {args.batch_size}", flush=True)
  material_hessian = None
  for energies, projection_method, model_name in group["energy_groups"]:
    print(f"  {model_name}: {len(energies)} energy type(s), projection_method={projection_method}", flush=True)
    term_hessian = differentiator().diff2(energies, targets, targets, projection_method=projection_method, dynamic_instances=True, separate_hessian_jacobian=True, grouped_add=True, auto_partition=False)
    material_hessian = term_hessian if material_hessian is None else material_hessian + term_hessian
  # Material connectivity is frozen per frame; ordinary scene contacts refresh
  # independently at every Newton iterate and every line-search trial.
  group["material_hessian"] = material_hessian

##################################################################
## Numerical buffers, CCD topology, and a fixed comparison camera
##################################################################
ndof = sum(att.size * att.correspondance.numInstances for att in targets)
initial_guess = gpuarray.zeros(ndof, np.float64)
# Keep MAS as the primary solver; this separate instance handles failed solves.
jacobi_solver = solver("jacobian")
assembled = matrix(ndof, ndof, symmetric_storage=True)
assembled.wrt = targets
gradient_segment_sizes = [att.size * att.correspondance.numInstances for att in targets]
diagonal_block_sizes = [att.size * att.size * att.correspondance.numInstances for att in targets]
assembled.gradient_segments_start_cpu = [0] + np.cumsum(gradient_segment_sizes).tolist()
assembled.diagonal_blocks_start_cpu = [0] + np.cumsum(diagonal_block_sizes).tolist()
assembled.diagonal = gpuarray.zeros(ndof, np.float64)
assembled.diagonal_blocks = gpuarray.zeros(sum(diagonal_block_sizes), np.float64)
assembled.diagonal_blocks_inverse = gpuarray.zeros(sum(diagonal_block_sizes), np.float64)
gradient = vector(ndof)
node_mass = gpuarray.zeros(len(GRID), np.float64)
node_momentum = gpuarray.zeros(3 * len(GRID), np.float64)
grid_target_values = gpuarray.zeros(3 * len(GRID), np.float64)
invalid = gpuarray.zeros(1, np.int32)
soft_velocity = gpuarray.to_gpu(np.tile(INITIAL_VELOCITY, 2 * nv))
affine_velocity = gpuarray.to_gpu(np.tile(INITIAL_VELOCITY, 2 * nv))
start_frame = 0
restart_time = 0.0
if args.restart:
  checkpoint = np.load(args.restart.resolve())
  if "grid_nodes" in checkpoint and int(checkpoint["grid_nodes"]) != GRID_N:
    raise ValueError("Restart checkpoint grid resolution does not match GRID_N")
  start_frame = int(checkpoint["next_frame"])
  restart_time = float(checkpoint["time"])
  update_position(soft_coordinates, checkpoint["soft_position"])
  for att, value in zip(affine_rows, checkpoint["affine_rows"]):
    att.updateValue(value, deepCopy=True)
  soft_velocity = gpuarray.to_gpu(checkpoint["soft_velocity"])
  affine_velocity = gpuarray.to_gpu(checkpoint["affine_velocity"])
  for group, prefix in zip(groups, ("liquid", "solid")):
    count_key = f"{prefix}_body_counts"
    if count_key in checkpoint and not np.array_equal(checkpoint[count_key], group["particle_counts"]):
      raise ValueError("Restart checkpoint per-body particle counts do not match the current configuration")
    if checkpoint[f"{prefix}_x"].size != 3 * group["count"]:
      raise ValueError("Restart checkpoint does not match the current number of MPM bunnies")
    group["state"] = {name: gpuarray.to_gpu(checkpoint[f"{prefix}_{name}"]) for name in ("x", "v", "F", "C")}
    if group["liquid"]:
      if "liquid_J" in checkpoint:
        liquid_J = checkpoint["liquid_J"]
      else:
        liquid_J = liquid_volume_from_deformation(checkpoint["liquid_F"])
        print(f"RESTART migrated liquid J from stored F at 80-digit precision: min={liquid_J.min():.9e} max={liquid_J.max():.9e} negative={np.count_nonzero(liquid_J < 0)}; no clamping/reset", flush=True)
      if liquid_J.size != group["count"] or not np.isfinite(liquid_J).all():
        raise ValueError("Restart liquid_J must be finite and match the particle count")
      group["state"]["J"] = gpuarray.to_gpu(np.ascontiguousarray(liquid_J, dtype=np.float64).ravel())
    group["frozen"]["x"], group["frozen"]["F"] = group["state"]["x"], group["state"]["F"]
    freeze_particle_model(group)
  print(f"RESTART loaded={args.restart.resolve()} next_frame={start_frame} time={restart_time}", flush=True)
# Initialize the frozen maps before evaluating particle geometry, without advancing.
for group in groups:
  state, frozen = group["state"], group["frozen"]
  transfer.get_function("p2g")(state["x"], state["v"], state["F"], state["C"], group["mass"], frozen["indices"], frozen["weights"], frozen["B"], node_mass, node_momentum, invalid, np.int32(group["count"]), np.int32(group["liquid"]), block=(128, 1, 1), grid=((group["count"] + 127) // 128, 1, 1))
  bind_particle_range(group)
if invalid.get()[0]:
  raise ValueError("Initial particle stencil left the grid")
initial_geometry = union_x.compute().value.copy()
ccd_capacity = 60_000_000
ccd = create_collision_detector(initial_geometry, faces, edges, query_ids, mesh_ids, ccd_capacity)
# Evaluate only the ordinary scene once to obtain its fixed sparse metadata;
# particle Hessians are external to this minimizer and are not evaluated here.
scene_hessian = s.minimizer.computeNumericValue()
frame_topology = FrameHierarchyTopology(scene_hessian, groups, args.batch_size, transfer)
hierarchy_frame = None
if not args.checkpoints_only:
  plotter = pv.Plotter(off_screen=not args.gui, window_size=(1280, 960))
  plotter.set_background("#e7ecf2")
  # Render exactly the same open-top mesh supplied to collision detection.
  render_container = pv.PolyData(container_points, np.column_stack([np.full(len(container_triangles), 3), container_triangles]).ravel())
  plotter.add_mesh(render_container, color="#8b9bab", opacity=0.18, show_edges=True)
  render_meshes = []
  for i in range(4):
    poly = pv.PolyData(positions[i][surface_ids], np.column_stack([np.full(len(render_triangles), 3), render_triangles]).ravel())
    plotter.add_mesh(poly, color=COLORS[i], smooth_shading=True)
    render_meshes.append(poly)
  for j, group in enumerate(groups):
    x = group["state"]["x"].get().reshape(-1, 3)
    group["render_meshes"] = []
    for k, body_id in enumerate(group["body_ids"]):
      first, last = group["body_offsets"][k:k + 2]
      poly = pv.PolyData(x[first:last])
      plotter.add_mesh(poly, color=COLORS[body_id], point_size=3, render_points_as_spheres=True)
      group["render_meshes"].append(poly)
  plotter.camera_position = [(1.15, 0.95, 1.30), (0.0, 0.32, 0.0), (0.0, 1.0, 0.0)]
  plotter.camera.parallel_projection = True
  plotter.camera.parallel_scale = 0.54
  plotter.add_text("Affine / FEM / liquid MPM / solid MPM", font_size=12, color="#253245", name="title")
  plotter.show(auto_close=False, interactive_update=True)
  plotter.screenshot(str(OUTPUT / "initial.jpg"))
statistics_path = OUTPUT / "statistics.json"
logs = json.loads(statistics_path.read_text())[:start_frame] if args.restart and statistics_path.exists() else []
physical_time = restart_time

##################################################################
## Forward frames: update support/P2G once, freeze it through all Newtons
##################################################################
for frame in range(start_frame, args.frames):
  frame_started = time.perf_counter()
  timing = {name: 0.0 for name in ["p2g", "hierarchy", "hierarchy_contact_cd", "assembly", "cg", "ccd", "cd", "ccd_sweep", "ccd_step", "line_search", "line_search_cd", "g2p"]}
  old_soft = soft_x.compute().value.copy()
  old_soft_values = [att.value.copy() for att in soft_coordinates]
  old_affine_rows = [att.value.copy() for att in affine_rows]
  old_affine = affine_x.compute().value.copy()
  node_mass.fill(0.0)
  node_momentum.fill(0.0)
  invalid.fill(0)
  tick = time.perf_counter()
  for group in groups:
    state, frozen = group["state"], group["frozen"]
    group["material_hessian"].clearDynamicCoordinateCache()
    frozen["x"], frozen["F"] = state["x"], state["F"]
    freeze_particle_model(group)
    transfer.get_function("p2g")(state["x"], state["v"], state["F"], state["C"], group["mass"], frozen["indices"], frozen["weights"], frozen["B"], node_mass, node_momentum, invalid, np.int32(group["count"]), np.int32(group["liquid"]), block=(128, 1, 1), grid=((group["count"] + 127) // 128, 1, 1))
    bind_particle_range(group)
  cuda.Context.synchronize()
  if invalid.get()[0]:
    raise RuntimeError(f"Frame {frame}: unsupported particle position; no stencil clamping")
  # Keep physical P2G mass for transfers; floor only the inertia coefficient.
  # gpuarray.maximum evaluates on the GPU, including all inactive grid nodes.
  grid_mass.updateValue(gpuarray.maximum(node_mass, np.float64(GRID_INERTIA_MIN_MASS)))
  cuda.Context.synchronize()
  timing["p2g"] = time.perf_counter() - tick
  # P2G and Hessian coordinates still refresh every frame. Only the MAS
  # preconditioner hierarchy is reused between scheduled rebuilds.
  tick = time.perf_counter()
  rebuild_hierarchy = hierarchy_frame is None or frame % HIERARCHY_REBUILD_INTERVAL == 0
  initial_contact_counts = []
  if rebuild_hierarchy and HIERARCHY_INCLUDE_INITIAL_CONTACTS:
    # New P2G maps use displacement from the fixed grid. Reset q before CD
    # so this is the committed start-of-frame geometry, not last frame's q.
    update_position(grid_coordinates, G.value)
    hierarchy_contact_started = time.perf_counter()
    ccd.cd(union_x.compute().value, CONTACT_DISTANCE**2)
    update_contacts(ccd, contacts, vertex_weight, transfer)
    cuda.Context.synchronize()
    timing["hierarchy_contact_cd"] = time.perf_counter() - hierarchy_contact_started
    initial_contact_counts = list(map(int, ccd.separated_counts))
  if rebuild_hierarchy:
    frame_topology.rebuild(s.minimizer.linearSolver, scene_hessian if HIERARCHY_INCLUDE_INITIAL_CONTACTS else None)
    hierarchy_frame = frame
  cuda.Context.synchronize()
  timing["hierarchy"] = time.perf_counter() - tick
  print(f"HIERARCHY frame={frame:03d} rebuilt={rebuild_hierarchy} source_frame={hierarchy_frame:03d} blocks={frame_topology.num_blocks} contact_blocks={frame_topology.contact_block_count} initial_contacts={initial_contact_counts} unique_particle_stencils={frame_topology.unique_stencil_counts} seconds={timing['hierarchy']:.6f}", flush=True)
  mass_host = node_mass.get()
  mpm_mass = sum(float(gpuarray.sum(group["mass"]).get()) for group in groups)
  mass_error = abs(mass_host.sum() - mpm_mass) / mpm_mass
  if mass_error > 1e-10:
    raise RuntimeError(f"P2G mass conservation failed: {mass_error}")
  h = FRAME_DT
  cg_iterations = newton_total = 0
  accepted_alpha = 1.0
  # One fixed-duration solve per frame; commit the final iterate at the Newton cap.
  update_position(grid_coordinates, G.value)
  for att, value in zip(soft_coordinates + affine_rows, old_soft_values + old_affine_rows):
    att.updateValue(value, deepCopy=True)
  h_attribute.updateValue([h])
  transfer.get_function("grid_targets")(node_mass, node_momentum, grid_target_values, np.float64(h), block=(128, 1, 1), grid=((len(GRID) + 127) // 128, 1, 1))
  qhat.updateValue(grid_target_values)
  gravity = gpuarray.to_gpu(np.tile([0.0, -9.8 * h * h, 0.0], 2 * nv))
  soft_target.updateValue(old_soft + h * soft_velocity + gravity)
  affine_target.updateValue(old_affine + h * affine_velocity + gravity)
  converged = False
  reference_gradient_norm = None
  for newton in range(MAX_NEWTON_ITERATIONS):
    newton_started = time.perf_counter()
    timing_before = timing.copy()
    newton_total += 1
    current_geometry = union_x.compute().value.copy()
    tick = time.perf_counter()
    ccd.cd(current_geometry, CONTACT_DISTANCE**2)
    update_contacts(ccd, contacts, vertex_weight, transfer)
    elapsed = time.perf_counter() - tick
    timing["ccd"] += elapsed
    timing["cd"] += elapsed
    energy_before = full_energy(s, groups)
    if not np.isfinite(energy_before):
      raise RuntimeError(f"Frame {frame}, Newton {newton}: nonfinite energy at current iterate")
    tick = time.perf_counter()
    assemble_batched(s.minimizer, groups, assembled, gradient, args.batch_size)
    cuda.Context.synchronize()
    timing["assembly"] += time.perf_counter() - tick
    free, _ = cuda.mem_get_info()
    peak_used = max(peak_used, gpu_total - free)
    tick = time.perf_counter()
    active_solver = s.minimizer.linearSolver
    mas_failure = None
    code = active_solver.computeSolution(assembled, gradient, initial_guess, tolerance=CG_TOLERANCE, maxIterations=MAX_CG_ITERATIONS, zero_initial_guess=True)
    mas_stats = active_solver.statistics.copy()
    # Stagnation keeps the MAS iterate; other failures, including local
    # inversion failure (-8), use Jacobi on the same Hessian and RHS.
    if code < 0 and code != -5:
      mas_failure = f"code={code}, reason={mas_stats.get('breakdown')}"
    mas_seconds = time.perf_counter() - tick
    if mas_failure is not None:
      print(f"JACOBI FALLBACK frame={frame:03d} k={newton:03d}: MAS failed ({mas_failure}); solving the same Hessian and RHS", flush=True)
      active_solver = jacobi_solver
      code = active_solver.computeSolution(assembled, gradient, initial_guess, tolerance=CG_TOLERANCE, maxIterations=MAX_CG_ITERATIONS, zero_initial_guess=True)
    timing["cg"] += time.perf_counter() - tick
    stats = active_solver.statistics.copy()
    use_gradient_direction = mas_stats.get("result") == -8 and code <= -1000 and stats.get("breakdown") == "non-positive curvature or invalid preconditioned residual"
    stats.update({"used_jacobi_fallback": mas_failure is not None, "used_gradient_fallback": use_gradient_direction, "mas_failure": mas_failure, "mas_attempt_seconds": mas_seconds})
    if mas_failure is not None:
      stats["mas_attempt"] = mas_stats
      cg_iterations += int(mas_stats.get("iterations", 0))
    cg_iterations += int(stats.get("iterations", 0))
    if code < 0:
      print(f"SOLVER WARNING solver={active_solver.solverName} frame={frame:03d} k={newton:03d} code={code} reason={stats.get('breakdown') or 'CG iteration limit'} iterations={stats.get('iterations')} relative_residual={stats.get('relative_residual')}; checking the returned iterate", flush=True)
    # A curvature breakdown preserves earlier CG updates. Prefer that partial
    # iterate if it gives a finite descent direction on the movable variables.
    use_partial_jacobi_direction = False
    if use_gradient_direction:
      direction = -active_solver.solution.copy()
      transfer.get_function("mask_inactive")(direction, node_mass, np.int32(SEPARATE_COORDINATES), block=(128, 1, 1), grid=((len(GRID) + 127) // 128, 1, 1))
      partial_slope = float(gpuarray.dot(gradient.value, direction).get())
      use_partial_jacobi_direction = np.isfinite(partial_slope) and partial_slope < 0.0
      use_gradient_direction = not use_partial_jacobi_direction
      if use_gradient_direction:
        print(f"GRADIENT FALLBACK frame={frame:03d} k={newton:03d}: no usable partial Jacobi iterate", flush=True)
        direction = -gradient.value.copy()
      else:
        print(f"PARTIAL JACOBI frame={frame:03d} k={newton:03d}: keeping the last descent iterate after breakdown at CG iteration {stats.get('iterations')}", flush=True)
    else:
      direction = -active_solver.solution.copy()
    stats.update({"used_gradient_fallback": use_gradient_direction, "used_partial_jacobi": bool(use_partial_jacobi_direction)})
    # Inactive physical grid nodes have a rest-position inertia target.
    # Keep them at rest if the approximate MAS solve leaves a small update.
    transfer.get_function("mask_inactive")(direction, node_mass, np.int32(SEPARATE_COORDINATES), block=(128, 1, 1), grid=((len(GRID) + 127) // 128, 1, 1))
    slope = float(gpuarray.dot(gradient.value, direction).get())
    gradient_inf = float(gpuarray.max(abs(gradient.value)).get())
    gradient_norm = float(gpuarray.dot(gradient.value, gradient.value).get())**0.5
    if reference_gradient_norm is None:
      reference_gradient_norm = max(gradient_norm, 1e-30)
    relative_gradient = gradient_norm / reference_gradient_norm
    relative_linear_residual = float(stats.get("relative_residual", np.nan))
    if not np.isfinite(slope):
      raise RuntimeError(f"Frame {frame}, Newton {newton}: {active_solver.solverName} PCG returned a non-finite gTd={slope}")
    originals = [att.value.copy() for att in targets]
    cursor = 0
    directions = []
    for att, original in zip(targets, originals):
      d = direction[cursor:cursor + original.size]
      cursor += original.size
      directions.append(d)
      att.updateValue(original + d, deepCopy=True)
    # Gather the full geometry direction for CCD without evaluating energy.
    full_geometry = union_x.compute().value.copy()
    for att, original in zip(targets, originals):
      att.updateValue(original, deepCopy=True)
    displacement = full_geometry - current_geometry
    motion = float(gpuarray.max(abs(displacement)).get()) / h
    # The same CCD sweep handles both body-body and body-container crossings.
    alpha = 1.0
    tick = time.perf_counter()
    ccd_direction = -displacement  # CCD uses old-new, not new-old.
    ccd, ccd_capacity = ccd_sweep_with_growth(ccd, ccd_capacity, current_geometry, CONTACT_DISTANCE**2, ccd_direction, alpha, faces, edges, query_ids, mesh_ids)
    elapsed = time.perf_counter() - tick
    timing["ccd"] += elapsed
    timing["ccd_sweep"] += elapsed
    tick = time.perf_counter()
    alpha = min(alpha, float(ccd.compute_largest_step_size(CCD_SLACKNESS, current_geometry, ccd_direction)))
    elapsed = time.perf_counter() - tick
    timing["ccd"] += elapsed
    timing["ccd_step"] += elapsed
    # Start gradient backtracking at half the CCD-safe step.
    if use_gradient_direction:
      alpha *= 0.5
    tick = time.perf_counter()
    accepted = False
    for backtrack in range(MAX_BACKTRACKS):
      for att, original, d in zip(targets, originals, directions):
        att.updateValue(original + alpha * d, deepCopy=True)
      trial_geometry = union_x.compute().value
      collision_started = time.perf_counter()
      ccd.cd(trial_geometry, CONTACT_DISTANCE**2)
      timing["line_search_cd"] += time.perf_counter() - collision_started
      update_contacts(ccd, contacts, vertex_weight, transfer)
      energy_after = full_energy(s, groups)
      if np.isfinite(energy_after) and energy_after <= energy_before + 1e-4 * alpha * slope:
        accepted = True
        break
      if backtrack + 1 < MAX_BACKTRACKS:
        alpha *= 0.5
    timing["line_search"] += time.perf_counter() - tick
    if not np.isfinite(energy_after):
      raise RuntimeError(f"Frame {frame}, Newton {newton}: every line-search trial had nonfinite energy")
    if not accepted:
      print(f"LINE SEARCH WARNING frame={frame:03d} k={newton:03d} accepting the final finite trial at alpha={alpha:.6e}", flush=True)
    accepted_alpha = alpha
    print(f"NEWTON frame={frame:03d} k={newton:03d} h={h:.6g} motion={motion:.6e} alpha={alpha:.6e} energy={energy_after:.9e} |g|inf={gradient_inf:.6e} relative_g={relative_gradient:.3e} linear_residual={relative_linear_residual:.3e} cg={stats.get('iterations')} ccd_candidates={ccd.candidate_count} contacts={ccd.separated_counts}", flush=True)
    # Retain per-Newton phases and native solver statistics for saved-frame benchmarks.
    if not args.checkpoints_only:
      with (OUTPUT / "newton_statistics.jsonl").open("a") as timing_file:
        timing_file.write(json.dumps({"frame": frame, "newton": newton, "seconds": time.perf_counter() - newton_started, "timing_seconds": {name: timing[name] - timing_before[name] for name in timing}, "solver": stats, "solver_code": code, "motion": motion, "alpha": alpha, "energy": energy_after, "candidates": ccd.candidate_count}) + "\n")
    # Neither a raw-gradient step nor a failed partial solve establishes convergence.
    if not use_gradient_direction and not use_partial_jacobi_direction and motion < MOTION_TOLERANCE:
      converged = True
      break

  accepted_at_newton_cap = not converged
  if accepted_at_newton_cap:
    print(f"FRAME CAP frame={frame:03d} accepting the last iterate after {MAX_NEWTON_ITERATIONS} Newton iterations motion={motion:.6e}", flush=True)


  ################################################################
  ## Commit once: APIC G2P, incremental liquid volume, and mesh velocities
  ################################################################
  tick = time.perf_counter()
  committed_states = []
  for group in groups:
    state, frozen = group["state"], group["frozen"]
    next_state = {key: gpuarray.empty_like(value) for key, value in state.items()}
    if not group["liquid"]:
      accepted_F = group["F"].compute().value.copy()
    transfer.get_function("g2p")(q.compute().value, np.float64(h), frozen["x"], frozen["F"], frozen["indices"], frozen["weights"], frozen["B"], next_state["x"], next_state["v"], next_state["F"], next_state["C"], np.int32(group["count"]), state["J"] if group["liquid"] else np.uintp(0), next_state["J"] if group["liquid"] else np.uintp(0), np.int32(group["liquid"]), block=(128, 1, 1), grid=((group["count"] + 127) // 128, 1, 1))
    if not group["liquid"]:
      error = float(gpuarray.max(abs(next_state["F"] - accepted_F)).get())
      if error > 1e-8:
        print(f"WARNING frame={frame:03d} material=solid: committed F differs from minimized F by {error:.6e}; continuing with G2P state", flush=True)
    committed_states.append(next_state)
  for group, next_state in zip(groups, committed_states):
    group["state"] = next_state
  soft_velocity = (soft_x.compute().value - old_soft) / h
  affine_velocity = (affine_x.compute().value - old_affine) / h
  physical_time += h
  cuda.Context.synchronize()
  timing["g2p"] = time.perf_counter() - tick

  ################################################################
  ## Save converged geometry/diagnostics and render one JPG per frame
  ################################################################
  if not args.checkpoints_only:
    soft_now = soft_x.compute().value.get().reshape(-1, 3)
    affine_now = affine_x.compute().value.get().reshape(-1, 3)
    for i in range(2):
      render_meshes[i].points = affine_now[i * nv:(i + 1) * nv][surface_ids]
      render_meshes[i + 2].points = soft_now[i * nv:(i + 1) * nv][surface_ids]
    for j, group in enumerate(groups):
      x = group["state"]["x"].get().reshape(-1, 3)
      for k, poly in enumerate(group["render_meshes"]):
        first, last = group["body_offsets"][k:k + 2]
        poly.points = x[first:last]
  particle_J = [float(gpuarray.min(group["state"]["J"]).get()) if group["liquid"] else float(np.linalg.det(group["state"]["F"].get().reshape(-1, 3, 3)).min()) for group in groups]
  soft_J = np.linalg.det(soft_F.compute().value.get().reshape(-1, 3, 3))
  affine_T_now = T.compute().value.get().reshape(2, 3, 4)
  A = affine_T_now[:, :, :3]
  # Dynamic computed buffers retain capacity; exclude their zero-filled tail.
  contact_min = min([float(gpuarray.min(record["distance"].compute().value[:record["primitive"].numInstances]).get()) for record in contacts if record["primitive"].numInstances] or [CONTACT_DISTANCE**2])**0.5
  # Diagnostic only: exclude fixed container vertices (which lie on the floor).
  body_geometry_now = union_x.compute().value.get().reshape(-1, 3)[body_query_ids]
  floor_clearance = float(body_geometry_now[:, 1].min() - container_points[:, 1].min())
  free, _ = cuda.mem_get_info()
  peak_used = max(peak_used, gpu_total - free)
  record = {"frame": frame, "time": physical_time, "h": h, "newton_iterations": newton_total, "cg_iterations": cg_iterations, "active_grid_nodes": int((mass_host > 0).sum()), "particles": 4 * args.particles, "accepted_alpha": accepted_alpha, "motion": motion, "gradient_inf": gradient_inf, "relative_gradient": relative_gradient, "true_linear_relative_residual": relative_linear_residual, "solver_relative_residual": stats.get("relative_residual"), "min_detF": float(min(*particle_J, soft_J.min())), "min_affine_det": float(np.linalg.det(A).min()), "affine_orthogonality_inf": float(np.abs(A.transpose(0, 2, 1) @ A - np.eye(3)).max()), "min_active_contact_distance": contact_min, "min_surface_particle_y_above_floor": floor_clearance, "p2g_relative_mass_error": mass_error, "timing_seconds": timing, "frame_seconds": time.perf_counter() - frame_started, "gpu_used_gib": (gpu_total - free) / 2**30, "gpu_sampled_peak_added_gib": (peak_used - (gpu_total - free_at_start)) / 2**30, "converged": converged, "accepted_at_newton_cap": accepted_at_newton_cap}
  record["particles"] = sum(group["count"] for group in groups)
  record["liquid_J_min"] = particle_J[0]
  record["liquid_J_max"] = float(gpuarray.max(groups[0]["state"]["J"]).get())
  record["hierarchy_rebuilt"] = rebuild_hierarchy
  record["hierarchy_source_frame"] = hierarchy_frame
  record["ccd_capacity"] = ccd_capacity
  logs.append(record)
  if not args.checkpoints_only:
    (OUTPUT / "statistics.json").write_text(json.dumps(logs, indent=2))
  print("FRAME " + json.dumps(record), flush=True)
  if not args.checkpoints_only:
    plotter.add_text(f"Frame {frame + 1}/{args.frames} | t={physical_time:.3f}s | MPM particles: {total_particles:,}", position="lower_left", font_size=12, color="#253245", name="status")
    plotter.update()
    plotter.screenshot(str(OUTPUT / f"frame_{frame:04d}.jpg"))
    np.savez_compressed(OUTPUT / f"frame_{frame:04d}.npz", time=physical_time, affine_T=affine_T_now, affine_surface=np.array([m.points for m in render_meshes[:2]]), soft_surface=np.array([m.points for m in render_meshes[2:]]), surface_triangles=render_triangles, liquid_positions=groups[0]["state"]["x"].get().reshape(-1, 3), solid_positions=groups[1]["state"]["x"].get().reshape(-1, 3), liquid_body_counts=groups[0]["particle_counts"], solid_body_counts=groups[1]["particle_counts"])
  # Checkpoint-only runs also save the first ten frames, including liquid J.
  if args.checkpoints_only or frame >= CHECKPOINT_START_FRAME:
    np.savez_compressed(OUTPUT / f"restart_after_frame_{frame:04d}.npz", completed_frame=frame, next_frame=frame + 1, time=physical_time, dt=h, grid_nodes=GRID_N, particles_per_bunny=args.particles, liquid_particles_per_bunny=args.liquid_particles, liquid_body_counts=groups[0]["particle_counts"], solid_body_counts=groups[1]["particle_counts"], soft_position=soft_x.value.get(), soft_velocity=soft_velocity.get(), affine_rows=np.array([att.value.get() for att in affine_rows]), affine_velocity=affine_velocity.get(), liquid_x=groups[0]["state"]["x"].get(), liquid_v=groups[0]["state"]["v"].get(), liquid_F=groups[0]["state"]["F"].get(), liquid_C=groups[0]["state"]["C"].get(), liquid_J=groups[0]["state"]["J"].get(), solid_x=groups[1]["state"]["x"].get(), solid_v=groups[1]["state"]["v"].get(), solid_F=groups[1]["state"]["F"].get(), solid_C=groups[1]["state"]["C"].get())
    print(f"CHECKPOINT saved={OUTPUT / f'restart_after_frame_{frame:04d}.npz'} next_frame={frame + 1}", flush=True)

video = None if args.checkpoints_only else save_video(OUTPUT, args.frames - start_frame, round(1 / FRAME_DT), first_frame=start_frame)
print(f"FINISHED elapsed={time.perf_counter() - started:.2f}s video={video}", flush=True)
ccd.close()
s.minimizer.linearSolver.reset()
jacobi_solver.reset()
if not args.checkpoints_only:
  plotter.close()
