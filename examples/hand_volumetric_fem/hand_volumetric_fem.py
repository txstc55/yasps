"""Forward FEM hand closure with affine bone cavities and optional bunny contact."""
from pathlib import Path
import glob
import json
import os
import sys
import time

import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

from yasps import scene
from helpers import TetrahedronStepLimiter, affine_determinant, affine_orthogonality, affine_target, apply_affine, closed_piece_vertex_masses, compact_triangles, edge_edge, encode_video, inertia, make_plotter, point_edge, point_point, point_triangle, save_surface_frame, stable_neo_hookean, tetrahedron_inversion_barrier, tetrahedron_signed_six_volumes, tetrahedron_vertex_masses, unique_edges, update_collision_pairs, validate_collision_pairs
from friction_helpers import closest_point_coord_and_tangent_basis_ee, closest_point_coord_and_tangent_basis_pe, closest_point_coord_and_tangent_basis_pp, closest_point_coord_and_tangent_basis_pt, friction_energy_ee, friction_energy_pe, friction_energy_pp, friction_energy_pt, lambda_last_h_ee, lambda_last_h_pe, lambda_last_h_pp, lambda_last_h_pt


##################################################################
## Settings: run directly for the empty hand. Set YASPS_HAND_BUNNY=1 for
## the otherwise identical 75 mm bunny-holding simulation.
##################################################################
EXAMPLE_DIR = Path(__file__).resolve().parent
os.chdir(EXAMPLE_DIR)
sys.path.insert(0, str(EXAMPLE_DIR.parent / "ccd"))
from ccd import CCD

INCLUDE_BUNNY = os.environ.get("YASPS_HAND_BUNNY", "0") == "1"
NUM_FRAMES = int(os.environ.get("YASPS_NUM_FRAMES", "200"))
DT_VALUE = 0.01
MOTION_TOLERANCE = 1.0e-2
CG_TOLERANCE = 1.0e-4
MAX_CG_ITERATIONS = 20000
MAX_LINE_SEARCH_STEPS = 8
MINIMUM_DEFORMATION_JACOBIAN = 1.0e-6
VOLUME_STEP_SAFETY = 0.9
VOLUME_BARRIER_ACTIVATION_JACOBIAN = 0.2
VOLUME_BARRIER_STIFFNESS_SCALE = 100000.0
GRAVITY_VALUE = 9.81
DHAT_VALUE = 2.0e-8
KAPPA_VALUE = 1.0e3
FRICTION_RATE = 0.5
TISSUE_YOUNG_VALUE = 30000.0
TISSUE_POISSON_VALUE = 0.475103734439834
TISSUE_DENSITY = 1060.0
BONE_DENSITY = 1900.0
BUNNY_YOUNG_VALUE = 300000.0
BUNNY_POISSON_VALUE = 0.45
BUNNY_DENSITY = 1100.0
AFFINE_MATRIX_TARGET_STIFFNESS = 100000.0
AFFINE_TRANSLATION_TARGET_STIFFNESS = 100000.0
AFFINE_RIGIDITY_STIFFNESS = 10000.0
SHOW_GUI = False
VIDEO_FPS = 100
VARIANT_NAME = "with_bunny" if INCLUDE_BUNNY else "empty_hand"
OUTPUT_DIR = EXAMPLE_DIR / "outputs" / VARIANT_NAME
FRAME_DIR = OUTPUT_DIR / "frames"
SURFACE_DIR = OUTPUT_DIR / "surface_npz"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FRAME_DIR.mkdir(exist_ok=True)
SURFACE_DIR.mkdir(exist_ok=True)
for old_path in glob.glob(str(FRAME_DIR / "frame_*.jpg")) + glob.glob(str(SURFACE_DIR / "frame_*.npz")):
  os.remove(old_path)


##################################################################
## Load the verified affine-first volume layout and compute physical mass.
##################################################################
asset = dict(np.load(EXAMPLE_DIR / "simulation_assets" / "hand_volumetric_fem_assets.npz", allow_pickle=False))
assert 1 <= NUM_FRAMES <= len(asset["times"])
assert float(asset["dt"]) == DT_VALUE
hand_rest = asset["hand_rest_positions"]
hand_tetrahedra = asset["hand_tetrahedra"]
bone_rest = asset["bone_rest_positions"]
free_rest = asset["free_rest_positions"]
bone_rest_homogeneous = asset["bone_rest_homogeneous"]
bone_vertex_to_body = asset["bone_vertex_to_affine_body"]
piece_targets = asset["piece_affine_targets"]
bone_count = len(asset["piece_names"])
bone_vertex_count = int(asset["bone_vertex_count"])
free_vertex_count = int(asset["free_vertex_count"])
hand_vertex_count = len(hand_rest)
tissue_vertex_masses, tissue_tet_masses = tetrahedron_vertex_masses(hand_rest, hand_tetrahedra, TISSUE_DENSITY)
physical_bone_masses, _ = closed_piece_vertex_masses(bone_rest, asset["bone_surface_triangles"], asset["piece_vertex_offsets"], asset["piece_triangle_offsets"], BONE_DENSITY)
hand_vertex_masses = tissue_vertex_masses.copy()
hand_vertex_masses[:bone_vertex_count] += physical_bone_masses
tissue_mu_value = TISSUE_YOUNG_VALUE / (2.0 * (1.0 + TISSUE_POISSON_VALUE))
tissue_lambda_value = TISSUE_YOUNG_VALUE * TISSUE_POISSON_VALUE / ((1.0 + TISSUE_POISSON_VALUE) * (1.0 - 2.0 * TISSUE_POISSON_VALUE))
tissue_bulk_modulus_value = tissue_lambda_value + 2.0 * tissue_mu_value / 3.0
tissue_volume_barrier_stiffness_value = VOLUME_BARRIER_STIFFNESS_SCALE * tissue_bulk_modulus_value

bunny_rest = asset["bunny_rest_positions"]
bunny_tetrahedra = asset["bunny_tetrahedra"]
bunny_vertex_count = len(bunny_rest)
bunny_vertex_masses, bunny_tet_masses = tetrahedron_vertex_masses(bunny_rest, bunny_tetrahedra, BUNNY_DENSITY)
bunny_mu_value = BUNNY_YOUNG_VALUE / (2.0 * (1.0 + BUNNY_POISSON_VALUE))
bunny_lambda_value = BUNNY_YOUNG_VALUE * BUNNY_POISSON_VALUE / ((1.0 + BUNNY_POISSON_VALUE) * (1.0 - 2.0 * BUNNY_POISSON_VALUE))
bunny_bulk_modulus_value = bunny_lambda_value + 2.0 * bunny_mu_value / 3.0
bunny_volume_barrier_stiffness_value = VOLUME_BARRIER_STIFFNESS_SCALE * bunny_bulk_modulus_value


##################################################################
## One hand mesh owns the 27 affine bodies, their cavity vertices, every
## independent tissue/skin vertex, the affine-first union, and all tets.
##################################################################
simulation = scene(f"hand_volumetric_{VARIANT_NAME}")
simulation.minimizer.setSolver("jacobian")
dt = simulation.addConstant("dt"); dt.updateValue(DT_VALUE)
gravity = simulation.addConstant("gravity"); gravity.updateValue(GRAVITY_VALUE)
dhat = simulation.addConstant("dhat"); dhat.updateValue(DHAT_VALUE)
kappa = simulation.addConstant("kappa"); kappa.updateValue(KAPPA_VALUE)
friction_rate = simulation.addConstant("friction_rate"); friction_rate.updateValue(FRICTION_RATE)
tissue_mu = simulation.addConstant("tissue_mu"); tissue_mu.updateValue(tissue_mu_value)
tissue_lambda = simulation.addConstant("tissue_lambda"); tissue_lambda.updateValue(tissue_lambda_value)
volume_barrier_activation = simulation.addConstant("volume_barrier_activation"); volume_barrier_activation.updateValue(VOLUME_BARRIER_ACTIVATION_JACOBIAN)
tissue_volume_barrier_stiffness = simulation.addConstant("tissue_volume_barrier_stiffness"); tissue_volume_barrier_stiffness.updateValue(tissue_volume_barrier_stiffness_value)

hand = simulation.addMesh("hand")
affine_bodies = hand.addPrimitive("affine_bodies", numInstances=bone_count)
affine = affine_bodies.addAttribute("affine", rows=3, cols=4); affine.updateValue(asset["rest_piece_affine"].ravel())
target_affine = affine_bodies.addConstant("target_affine", rows=3, cols=4); target_affine.updateValue(piece_targets[0].ravel())
matrix_target_weight = affine_bodies.addConstant("matrix_target_weight"); matrix_target_weight.updateValue(np.full(bone_count, AFFINE_MATRIX_TARGET_STIFFNESS))
translation_target_weight = affine_bodies.addConstant("translation_target_weight"); translation_target_weight.updateValue(np.full(bone_count, AFFINE_TRANSLATION_TARGET_STIFFNESS))
rigidity_weight = affine_bodies.addConstant("rigidity_weight"); rigidity_weight.updateValue(np.full(bone_count, AFFINE_RIGIDITY_STIFFNESS))

bone_vertices = hand.addPrimitive("bone_vertices", numInstances=bone_vertex_count)
bone_rest_position = bone_vertices.addConstant("rest_position", rows=3, cols=1); bone_rest_position.updateValue(bone_rest.ravel())
bone_rest_homogeneous_attribute = bone_vertices.addConstant("rest_homogeneous", rows=4, cols=1); bone_rest_homogeneous_attribute.updateValue(bone_rest_homogeneous.ravel())
bone_last_position = bone_vertices.addConstant("last_position", rows=3, cols=1); bone_last_position.updateValue(bone_rest.ravel())
bone_velocity = bone_vertices.addConstant("velocity", rows=3, cols=1); bone_velocity.updateValue(np.zeros(bone_vertex_count * 3))
bone_mass = bone_vertices.addConstant("mass"); bone_mass.updateValue(hand_vertex_masses[:bone_vertex_count])
bone_to_affine = bone_vertices.addConnectivity("to_affine_body", affine_bodies, bone_vertex_to_body, 1)
joined_affine = bone_vertices.addAttribute("joined_affine", through=bone_to_affine, source=affine).resize(3, 4)
bone_position = bone_vertices.addAttribute("position", computed_attribute=joined_affine * bone_rest_homogeneous_attribute)

muscle_vertices = hand.addPrimitive("muscle_vertices", numInstances=free_vertex_count)
muscle_position = muscle_vertices.addAttribute("position", rows=3, cols=1); muscle_position.updateValue(free_rest.ravel())
muscle_rest_position = muscle_vertices.addConstant("rest_position", rows=3, cols=1); muscle_rest_position.updateValue(free_rest.ravel())
muscle_last_position = muscle_vertices.addConstant("last_position", rows=3, cols=1); muscle_last_position.updateValue(free_rest.ravel())
muscle_velocity = muscle_vertices.addConstant("velocity", rows=3, cols=1); muscle_velocity.updateValue(np.zeros(free_vertex_count * 3))
muscle_mass = muscle_vertices.addConstant("mass"); muscle_mass.updateValue(hand_vertex_masses[bone_vertex_count:])

hand_vertices = hand.addPrimitiveUnion("hand_vertices", [bone_vertices, muscle_vertices])
hand_position = hand_vertices.addAttribute("position")
hand_rest_position = hand_vertices.addAttribute("rest_position")
hand_last_position = hand_vertices.addAttribute("last_position")
hand_velocity = hand_vertices.addAttribute("velocity")
hand_mass = hand_vertices.addAttribute("mass")

hand_tets = hand.addPrimitive("tetrahedra", numInstances=len(hand_tetrahedra))
hand_tets_to_vertices = hand_tets.addConnectivity("to_vertices", hand_vertices, hand_tetrahedra, 4)
hand_tet_position = hand_tets.addAttribute("position", through=hand_tets_to_vertices, source=hand_position)
hand_tet_rest_position = hand_tets.addAttribute("rest_position", through=hand_tets_to_vertices, source=hand_rest_position)

# A one-to-one gather exposes a 3x3 inner inertia Hessian. Bone branches then
# multiply by only one 3x12 affine Jacobian; free branches remain identity.
hand_inertia_terms = hand.addPrimitive("inertia_terms", numInstances=hand_vertex_count)
hand_inertia_to_vertices = hand_inertia_terms.addConnectivity("to_vertices", hand_vertices, np.arange(hand_vertex_count, dtype=np.uint32), 1)
hand_inertia_position = hand_inertia_terms.addAttribute("position", through=hand_inertia_to_vertices, source=hand_position).resize(3, 1)
hand_inertia_last = hand_inertia_terms.addAttribute("last_position", through=hand_inertia_to_vertices, source=hand_last_position).resize(3, 1)
hand_inertia_velocity = hand_inertia_terms.addAttribute("velocity", through=hand_inertia_to_vertices, source=hand_velocity).resize(3, 1)
hand_inertia_mass = hand_inertia_terms.addAttribute("mass", through=hand_inertia_to_vertices, source=hand_mass).resize(1, 1)


##################################################################
## The optional bunny is an ordinary independent volumetric FEM mesh.
##################################################################
bunny = None
bunny_vertices = None
bunny_position = None
bunny_last_position = None
bunny_velocity = None
if INCLUDE_BUNNY:
  bunny_mu = simulation.addConstant("bunny_mu"); bunny_mu.updateValue(bunny_mu_value)
  bunny_lambda = simulation.addConstant("bunny_lambda"); bunny_lambda.updateValue(bunny_lambda_value)
  bunny_volume_barrier_stiffness = simulation.addConstant("bunny_volume_barrier_stiffness"); bunny_volume_barrier_stiffness.updateValue(bunny_volume_barrier_stiffness_value)
  bunny = simulation.addMesh("bunny")
  bunny_vertices = bunny.addPrimitive("vertices", numInstances=bunny_vertex_count)
  bunny_position = bunny_vertices.addAttribute("position", rows=3, cols=1); bunny_position.updateValue(bunny_rest.ravel())
  bunny_rest_position = bunny_vertices.addConstant("rest_position", rows=3, cols=1); bunny_rest_position.updateValue(bunny_rest.ravel())
  bunny_last_position = bunny_vertices.addConstant("last_position", rows=3, cols=1); bunny_last_position.updateValue(bunny_rest.ravel())
  bunny_velocity = bunny_vertices.addConstant("velocity", rows=3, cols=1); bunny_velocity.updateValue(np.zeros(bunny_vertex_count * 3))
  bunny_mass = bunny_vertices.addConstant("mass"); bunny_mass.updateValue(bunny_vertex_masses)
  bunny_tets = bunny.addPrimitive("tetrahedra", numInstances=len(bunny_tetrahedra))
  bunny_tets_to_vertices = bunny_tets.addConnectivity("to_vertices", bunny_vertices, bunny_tetrahedra, 4)
  bunny_tet_position = bunny_tets.addAttribute("position", through=bunny_tets_to_vertices, source=bunny_position)
  bunny_tet_rest_position = bunny_tets.addAttribute("rest_position", through=bunny_tets_to_vertices, source=bunny_rest_position)


##################################################################
## The union retains every volume vertex so its global indices agree with the
## tetrahedra and YASPS connectivity. CCD separately receives only the outer
## skin, all 27 cavity surfaces, and the optional bunny boundary.
##################################################################
collisions = simulation.addMesh("collisions")
collision_children = [hand_vertices] + ([bunny_vertices] if INCLUDE_BUNNY else [])
collision_vertices = collisions.addPrimitiveUnion("vertices", collision_children)
collision_position = collision_vertices.addAttribute("position")
collision_targets = [affine, muscle_position] + ([bunny_position] if INCLUDE_BUNNY else [])
friction_last_position = None
if INCLUDE_BUNNY:
  friction_last_position = collision_vertices.addConstant("friction_last_position", rows=3, cols=1)
  friction_last_position.updateValue(np.concatenate((hand_rest, bunny_rest)).ravel())
collision_primitives = []
collision_connectivities = []
for name, width, barrier in (("pp", 2, point_point), ("pe", 3, point_edge), ("pt", 4, point_triangle), ("ee", 4, edge_edge)):
  primitive = collisions.addPrimitive(name, numInstances=0, isDynamic=True)
  connectivity = primitive.addConnectivity("to_vertices", collision_vertices, [], width)
  positions = primitive.addAttribute("position", through=connectivity, source=collision_position)
  energy = primitive.addAttribute("energy", computed_attribute=barrier(positions, dhat, kappa))
  simulation.addEnergy(energy, targets=collision_targets, dynamic_instances=True, projection_method=2, separate_hessian_jacobian=True)
  collision_primitives.append(primitive)
  collision_connectivities.append(connectivity)

# Friction is lagged: its contact set, coordinates, tangents, and normal loads
# come from the previous converged frame and remain fixed during Newton.
friction_primitives = []
friction_connectivities = []
if INCLUDE_BUNNY:
  friction_terms = (("pp", 2, closest_point_coord_and_tangent_basis_pp, lambda_last_h_pp, friction_energy_pp), ("pe", 3, closest_point_coord_and_tangent_basis_pe, lambda_last_h_pe, friction_energy_pe), ("pt", 4, closest_point_coord_and_tangent_basis_pt, lambda_last_h_pt, friction_energy_pt), ("ee", 4, closest_point_coord_and_tangent_basis_ee, lambda_last_h_ee, friction_energy_ee))
  for name, width, closest, normal_force, friction in friction_terms:
    primitive = collisions.addPrimitive(f"{name}_friction", numInstances=0, isDynamic=True)
    connectivity = primitive.addConnectivity("to_vertices", collision_vertices, [], width)
    positions = primitive.addAttribute("position", through=connectivity, source=collision_position)
    old_positions = primitive.addAttribute("last_position", through=connectivity, source=friction_last_position)
    coordinate, tangent_basis = closest(old_positions)
    normal_load = normal_force(old_positions, coordinate, dhat, kappa)
    energy = primitive.addAttribute("energy", computed_attribute=friction(positions, old_positions, dhat, dt, friction_rate, coordinate, tangent_basis.row(0), tangent_basis.row(1), normal_load))
    simulation.addEnergy(energy, targets=collision_targets, dynamic_instances=True, projection_method=2, separate_hessian_jacobian=True)
    friction_primitives.append(primitive)
    friction_connectivities.append(connectivity)


##################################################################
## Register trajectory, affine-rigidity, inertia, and elastic energies.
##################################################################
trajectory_energy = affine_bodies.addAttribute("trajectory_energy", computed_attribute=affine_target(affine, target_affine, matrix_target_weight, translation_target_weight, dt))
orthogonality_energy = affine_bodies.addAttribute("orthogonality_energy", computed_attribute=affine_orthogonality(affine, rigidity_weight, dt))
determinant_energy = affine_bodies.addAttribute("determinant_energy", computed_attribute=affine_determinant(affine, rigidity_weight, dt))
hand_inertia_energy = hand_inertia_terms.addAttribute("energy", computed_attribute=inertia(hand_inertia_last, hand_inertia_velocity, dt, hand_inertia_position, hand_inertia_mass, gravity))
hand_elastic_energy = hand_tets.addAttribute("energy", computed_attribute=stable_neo_hookean(hand_tet_rest_position, hand_tet_position, tissue_mu, tissue_lambda, dt))
hand_inversion_energy = hand_tets.addAttribute("inversion_energy", computed_attribute=tetrahedron_inversion_barrier(hand_tet_rest_position, hand_tet_position, tissue_volume_barrier_stiffness, volume_barrier_activation, dt))
simulation.addEnergy(trajectory_energy, projection_method=-1)
simulation.addEnergy(orthogonality_energy, projection_method=1)
simulation.addEnergy(determinant_energy, projection_method=1)
simulation.addEnergy(hand_inertia_energy, projection_method=-1, separate_hessian_jacobian=True)
simulation.addEnergy(hand_elastic_energy, projection_method=1, separate_hessian_jacobian=True)
simulation.addEnergy(hand_inversion_energy, projection_method=2, separate_hessian_jacobian=True)
if INCLUDE_BUNNY:
  bunny_inertia_energy = bunny_vertices.addAttribute("inertia_energy", computed_attribute=inertia(bunny_last_position, bunny_velocity, dt, bunny_position, bunny_mass, gravity))
  bunny_elastic_energy = bunny_tets.addAttribute("energy", computed_attribute=stable_neo_hookean(bunny_tet_rest_position, bunny_tet_position, bunny_mu, bunny_lambda, dt))
  bunny_inversion_energy = bunny_tets.addAttribute("inversion_energy", computed_attribute=tetrahedron_inversion_barrier(bunny_tet_rest_position, bunny_tet_position, bunny_volume_barrier_stiffness, volume_barrier_activation, dt))
  simulation.addEnergy(bunny_inertia_energy, projection_method=-1)
  simulation.addEnergy(bunny_elastic_energy, projection_method=1)
  simulation.addEnergy(bunny_inversion_energy, projection_method=2)

targets = [affine, muscle_position] + ([bunny_position] if INCLUDE_BUNNY else [])
simulation.addMinimizeTarget(targets)


##################################################################
## Verify native 3x4 affine replay and initialize component-aware CCD.
##################################################################
maximum_native_affine_error = 0.0
for verification_frame in (0, 49, 99, 149, 199):
  affine.updateValue(piece_targets[verification_frame].ravel())
  actual = bone_position.compute().value.get().reshape(-1, 3)
  expected = apply_affine(bone_rest_homogeneous, bone_vertex_to_body, piece_targets[verification_frame])
  error = float(np.max(np.abs(actual - expected)))
  maximum_native_affine_error = max(maximum_native_affine_error, error)
  np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12)
affine.updateValue(asset["rest_piece_affine"].ravel())

collision_triangle_parts = [asset["hand_boundary_triangles"]]
collision_surface_parts = [asset["hand_surface_indices"]]
mesh_id_parts = [asset["hand_mesh_ids"]]
if INCLUDE_BUNNY:
  collision_triangle_parts.append(asset["bunny_surface_triangles"] + hand_vertex_count)
  collision_surface_parts.append(asset["bunny_surface_indices"] + hand_vertex_count)
  mesh_id_parts.append(np.zeros(bunny_vertex_count, dtype=np.uint32))
collision_triangles = np.vstack(collision_triangle_parts).astype(np.uint32)
collision_surface_indices = np.concatenate(collision_surface_parts).astype(np.uint32)
collision_edges = unique_edges(collision_triangles)
collision_mesh_ids = np.concatenate(mesh_id_parts).astype(np.uint32)
collision_vertex_count = hand_vertex_count + (bunny_vertex_count if INCLUDE_BUNNY else 0)
collision_surface_mask = np.zeros(collision_vertex_count, dtype=bool); collision_surface_mask[collision_surface_indices] = True
np.testing.assert_array_equal(np.unique(collision_triangles), collision_surface_indices)
np.testing.assert_array_equal(asset["hand_surface_indices"], np.arange(len(asset["hand_surface_indices"]), dtype=np.uint32))
assert not np.any(collision_surface_mask[len(asset["hand_surface_indices"]):hand_vertex_count])
assert np.all(collision_surface_mask[collision_triangles])
assert np.all(collision_mesh_ids[asset["outer_surface_indices"]] == 0)
assert np.all(collision_mesh_ids[:bone_vertex_count] == 2 + asset["bone_vertex_piece_ids"])
assert np.all(collision_mesh_ids[collision_triangles] == collision_mesh_ids[collision_triangles][:, :1])
if INCLUDE_BUNNY:
  assert np.all(collision_mesh_ids[hand_vertex_count:] == 0)
initial_collision_position = collision_position.compute().value
np.testing.assert_allclose(initial_collision_position.get().reshape(-1, 3)[:hand_vertex_count], hand_rest, rtol=0.0, atol=1.0e-12)
detector = CCD(len(collision_surface_indices), collision_vertex_count, max_cd_pairs=10000000, max_ccd_pairs=30000000, mesh_indices=collision_mesh_ids, print_timings=False)
detector.init_faces(initial_collision_position, gpuarray.to_gpu(collision_triangles.ravel()), gpuarray.to_gpu(collision_surface_indices), len(collision_triangles))
detector.init_edges(initial_collision_position, initial_collision_position, gpuarray.to_gpu(collision_edges.ravel()), len(collision_edges))
initial_collision_counts = update_collision_pairs(detector, initial_collision_position, DHAT_VALUE, collision_primitives, collision_connectivities)
validate_collision_pairs(detector, initial_collision_counts, collision_vertex_count, collision_surface_mask, collision_mesh_ids)
hand_step_limiter = TetrahedronStepLimiter(hand_rest, hand_tetrahedra)
bunny_step_limiter = TetrahedronStepLimiter(bunny_rest, bunny_tetrahedra, vertex_offset=hand_vertex_count) if INCLUDE_BUNNY else None


##################################################################
## Fixed camera, compact surface state, settings, and frame zero.
##################################################################
gpu_free_at_start, gpu_total = cuda.mem_get_info()
plotter, outer_surface, bone_surface, bunny_surface = make_plotter(asset, INCLUDE_BUNNY, not SHOW_GUI, (1600, 1000))
title = "Volumetric hand holding a FEM bunny" if INCLUDE_BUNNY else "Volumetric hand driven by affine bone cavities"
plotter.add_text(title, position="upper_left", font_size=14, color="#263449", name="title")
plotter.add_text(f"Frame 001/{NUM_FRAMES}", position="lower_left", font_size=13, color="#263449", name="frame")
settings = {"variant": VARIANT_NAME, "frames": NUM_FRAMES, "dt": DT_VALUE, "motion_tolerance_m_per_s": MOTION_TOLERANCE, "dhat_squared_m2": DHAT_VALUE, "minimum_deformation_jacobian": MINIMUM_DEFORMATION_JACOBIAN, "volume_step_safety": VOLUME_STEP_SAFETY, "volume_barrier_activation_jacobian": VOLUME_BARRIER_ACTIVATION_JACOBIAN, "volume_barrier_stiffness_scale": VOLUME_BARRIER_STIFFNESS_SCALE, "tissue_volume_barrier_stiffness_pa": tissue_volume_barrier_stiffness_value, "affine_bodies": bone_count, "bone_surface_vertices": bone_vertex_count, "free_tissue_vertices": free_vertex_count, "hand_tetrahedra": len(hand_tetrahedra), "outer_skin_vertices": len(asset["outer_surface_indices"]), "collision_surface_vertices": len(collision_surface_indices), "collision_surface_triangles": len(collision_triangles), "tissue_young_pa": TISSUE_YOUNG_VALUE, "tissue_poisson": TISSUE_POISSON_VALUE, "tissue_density_kg_m3": TISSUE_DENSITY, "tissue_mass_kg": float(tissue_vertex_masses.sum()), "bone_density_kg_m3": BONE_DENSITY, "bone_mass_kg": float(physical_bone_masses.sum()), "bunny_enabled": INCLUDE_BUNNY, "bunny_mass_kg": float(bunny_vertex_masses.sum()) if INCLUDE_BUNNY else 0.0, "bunny_volume_barrier_stiffness_pa": bunny_volume_barrier_stiffness_value if INCLUDE_BUNNY else 0.0, "friction_coefficient": FRICTION_RATE if INCLUDE_BUNNY else 0.0, "native_affine_replay_max_error_m": maximum_native_affine_error}
(OUTPUT_DIR / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
hand_cpu = hand_position.compute().value.get().reshape(-1, 3)
bunny_cpu = bunny_position.value.get().reshape(-1, 3) if INCLUDE_BUNNY else None
save_surface_frame(SURFACE_DIR / "frame_0000.npz", 0, 0.0, asset, hand_cpu, asset["rest_piece_affine"], piece_targets[0], bunny_cpu)
outer_surface.points = hand_cpu[asset["outer_surface_indices"]]
bone_surface.points = hand_cpu[:bone_vertex_count]
if INCLUDE_BUNNY:
  bunny_surface.points = bunny_cpu[asset["bunny_surface_indices"]]
plotter.render()
plotter.screenshot(str(FRAME_DIR / "frame_0000.jpg"))


##################################################################
## Implicit-Euler forward solve. The CCD sweep uses every actual vertex,
## including affine cavity vertices, but only declared boundary topology.
##################################################################
statistics = []
simulation_start = time.perf_counter()
for frame in range(1, NUM_FRAMES):
  frame_start = time.perf_counter()
  target_affine.updateValue(piece_targets[frame].ravel())
  bone_last_position.updateValue(bone_position.compute().value, deepCopy=True)
  muscle_last_position.updateValue(muscle_position.value, deepCopy=True)
  if INCLUDE_BUNNY:
    bunny_last_position.updateValue(bunny_position.value, deepCopy=True)
    friction_last_position.updateValue(collision_position.compute().value, deepCopy=True)
    friction_counts = update_collision_pairs(detector, friction_last_position.value, DHAT_VALUE, friction_primitives, friction_connectivities)
    validate_collision_pairs(detector, friction_counts, collision_vertex_count, collision_surface_mask, collision_mesh_ids)
  else:
    friction_counts = (0, 0, 0, 0)

  newton_iteration = 0
  cg_iterations = 0
  line_search_reductions = 0
  solve_seconds = 0.0
  minimum_ccd_step = 1.0
  minimum_volume_step = 1.0
  minimum_trial_jacobian = float("inf")
  volume_limited_steps = 0
  while True:
    union_before = collision_position.compute().value.copy()
    collision_counts = update_collision_pairs(detector, union_before, DHAT_VALUE, collision_primitives, collision_connectivities)
    validate_collision_pairs(detector, collision_counts, collision_vertex_count, collision_surface_mask, collision_mesh_ids)
    energy_before = simulation.computeTotalEnergy()
    solve_start = time.perf_counter()
    exit_code = simulation.minimizer.computeHessianAndGradient(tolerance=CG_TOLERANCE, maxIterations=MAX_CG_ITERATIONS)
    solve_seconds += time.perf_counter() - solve_start
    if exit_code < 0:
      raise RuntimeError(f"GPU solver failed at frame {frame}, Newton {newton_iteration}: {exit_code}; {simulation.minimizer.linearSolver.statistics}")
    directions = simulation.minimizer.solutionSegments
    if len(directions) != len(targets):
      raise RuntimeError("The minimizer returned the wrong number of solution segments.")
    cg_iterations += int(simulation.minimizer.linearSolver.statistics.get("iterations", 0))
    originals = [target.value.copy() for target in targets]
    for target, original, direction in zip(targets, originals, directions):
      target.updateValue(original - direction, deepCopy=True)
    union_after_full_step = collision_position.compute().value
    union_direction = union_before - union_after_full_step
    maximum_motion = float(gpuarray.max(abs(union_direction)).get()) / DT_VALUE
    for target, original in zip(targets, originals):
      target.updateValue(original, deepCopy=True)

    sweep_alpha = 1.0
    for sweep_attempt in range(MAX_LINE_SEARCH_STEPS):
      try:
        detector.ccd(union_before, DHAT_VALUE, union_direction, sweep_alpha)
        break
      except OverflowError as error:
        if "broad-phase candidate capacity" not in str(error):
          raise
        sweep_alpha *= 0.5
    else:
      for target, original in zip(targets, originals):
        target.updateValue(original, deepCopy=True)
      raise RuntimeError(f"CCD broad phase stayed over capacity at frame {frame}, Newton {newton_iteration}.")
    ccd_step_size = float(detector.compute_largest_step_size(0.5, union_before, union_direction))
    minimum_ccd_step = min(minimum_ccd_step, ccd_step_size)
    step_size = hand_step_limiter.compute_largest_step_size(union_before, union_direction, ccd_step_size, MINIMUM_DEFORMATION_JACOBIAN, VOLUME_STEP_SAFETY)
    if INCLUDE_BUNNY:
      step_size = bunny_step_limiter.compute_largest_step_size(union_before, union_direction, step_size, MINIMUM_DEFORMATION_JACOBIAN, VOLUME_STEP_SAFETY)
    if step_size <= 0.0:
      current_jacobian = min(hand_step_limiter.minimum_jacobian(union_before), bunny_step_limiter.minimum_jacobian(union_before) if INCLUDE_BUNNY else float("inf"))
      raise RuntimeError(f"The current state is not volume-feasible at frame {frame}, Newton {newton_iteration}: minimum det(F)={current_jacobian:.9e}.")
    minimum_volume_step = min(minimum_volume_step, step_size)
    volume_limited_steps += int(step_size < ccd_step_size * (1.0 - 1.0e-12))
    for line_search_iteration in range(MAX_LINE_SEARCH_STEPS):
      for target, original, direction in zip(targets, originals, directions):
        target.updateValue(original - step_size * direction, deepCopy=True)
      trial_union = collision_position.compute().value
      trial_jacobian = min(hand_step_limiter.minimum_jacobian(trial_union), bunny_step_limiter.minimum_jacobian(trial_union) if INCLUDE_BUNNY else float("inf"))
      minimum_trial_jacobian = min(minimum_trial_jacobian, trial_jacobian)
      if not np.isfinite(trial_jacobian) or trial_jacobian <= MINIMUM_DEFORMATION_JACOBIAN:
        step_size *= 0.5
        continue
      collision_counts = update_collision_pairs(detector, trial_union, DHAT_VALUE, collision_primitives, collision_connectivities, cached_alpha=step_size)
      validate_collision_pairs(detector, collision_counts, collision_vertex_count, collision_surface_mask, collision_mesh_ids)
      energy_after = simulation.computeTotalEnergy()
      if np.isfinite(energy_after) and energy_after <= energy_before + 1.0e-12 * max(1.0, abs(energy_before)):
        break
      step_size *= 0.5
    else:
      for target, original in zip(targets, originals):
        target.updateValue(original, deepCopy=True)
      update_collision_pairs(detector, collision_position.compute().value, DHAT_VALUE, collision_primitives, collision_connectivities)
      raise RuntimeError(f"Line search failed at frame {frame}, Newton {newton_iteration}.")
    newton_iteration += 1
    line_search_reductions += line_search_iteration
    print(f"frame={frame:03d} newton={newton_iteration:03d} motion={maximum_motion:.6e} step={step_size:.6e} minJ={trial_jacobian:.6e} cg={simulation.minimizer.linearSolver.statistics.get('iterations', 0)} pairs={collision_counts} energy={energy_before:.9e}->{energy_after:.9e}", flush=True)
    if maximum_motion < MOTION_TOLERANCE:
      break

  bone_velocity.updateValue((bone_position.compute().value - bone_last_position.value) * (1.0 / DT_VALUE), deepCopy=True)
  muscle_velocity.updateValue((muscle_position.value - muscle_last_position.value) * (1.0 / DT_VALUE), deepCopy=True)
  if INCLUDE_BUNNY:
    bunny_velocity.updateValue((bunny_position.value - bunny_last_position.value) * (1.0 / DT_VALUE), deepCopy=True)

  hand_cpu = hand_position.compute().value.get().reshape(-1, 3)
  bunny_cpu = bunny_position.value.get().reshape(-1, 3) if INCLUDE_BUNNY else None
  current_affine = affine.value.get().reshape(bone_count, 3, 4)
  exact_bones = apply_affine(bone_rest_homogeneous, bone_vertex_to_body, piece_targets[frame])
  bone_target_error = float(np.max(np.abs(hand_cpu[:bone_vertex_count] - exact_bones)))
  hand_det_f = tetrahedron_signed_six_volumes(hand_cpu, hand_tetrahedra) / (6.0 * tissue_tet_masses / TISSUE_DENSITY)
  worst_hand_tet = int(np.argmin(hand_det_f))
  bunny_minimum_det_f = None
  bunny_inverted_tet_count = 0
  if INCLUDE_BUNNY:
    bunny_det_f = tetrahedron_signed_six_volumes(bunny_cpu, bunny_tetrahedra) / (6.0 * bunny_tet_masses / BUNNY_DENSITY)
    bunny_minimum_det_f = float(bunny_det_f.min())
    bunny_inverted_tet_count = int(np.count_nonzero(bunny_det_f <= 0.0))
  gpu_free, _ = cuda.mem_get_info()
  frame_record = {"frame": frame, "time": frame * DT_VALUE, "newton_iterations": newton_iteration, "cg_iterations": cg_iterations, "solve_seconds": solve_seconds, "frame_seconds": time.perf_counter() - frame_start, "line_search_reductions": line_search_reductions, "minimum_ccd_step": minimum_ccd_step, "minimum_volume_step": minimum_volume_step, "minimum_trial_jacobian": minimum_trial_jacobian, "volume_limited_steps": volume_limited_steps, "maximum_motion_m_per_s": maximum_motion, "collision_pairs": collision_counts, "friction_pairs": friction_counts, "bone_target_max_error_m": bone_target_error, "hand_minimum_det_f": float(hand_det_f[worst_hand_tet]), "hand_inverted_tet_count": int(np.count_nonzero(hand_det_f <= 0.0)), "worst_hand_tet": worst_hand_tet, "worst_hand_tet_indices": hand_tetrahedra[worst_hand_tet].tolist(), "worst_hand_tet_affine_vertices": int(np.count_nonzero(hand_tetrahedra[worst_hand_tet] < bone_vertex_count)), "bunny_minimum_det_f": bunny_minimum_det_f, "bunny_inverted_tet_count": bunny_inverted_tet_count, "gpu_memory_gib": (gpu_total - gpu_free) / 2**30, "gpu_memory_added_gib": (gpu_free_at_start - gpu_free) / 2**30}
  statistics.append(frame_record)
  save_surface_frame(SURFACE_DIR / f"frame_{frame:04d}.npz", frame, frame * DT_VALUE, asset, hand_cpu, current_affine, piece_targets[frame], bunny_cpu)
  outer_surface.points = hand_cpu[asset["outer_surface_indices"]]
  bone_surface.points = hand_cpu[:bone_vertex_count]
  outer_surface.compute_normals(cell_normals=False, point_normals=True, inplace=True)
  bone_surface.compute_normals(cell_normals=False, point_normals=True, inplace=True)
  if INCLUDE_BUNNY:
    bunny_surface.points = bunny_cpu[asset["bunny_surface_indices"]]
    bunny_surface.compute_normals(cell_normals=False, point_normals=True, inplace=True)
  plotter.add_text(f"Frame {frame + 1:03d}/{NUM_FRAMES}", position="lower_left", font_size=13, color="#263449", name="frame")
  plotter.render()
  if SHOW_GUI:
    plotter.update()
  plotter.screenshot(str(FRAME_DIR / f"frame_{frame:04d}.jpg"))
  (OUTPUT_DIR / "statistics.json").write_text(json.dumps({"settings": settings, "frames": statistics}, indent=2) + "\n")
  print(f"FRAME_DONE {json.dumps(frame_record)}", flush=True)


##################################################################
## Encode exactly 200 frames at the physical 100 fps playback rate.
##################################################################
video_path = OUTPUT_DIR / f"hand_volumetric_{VARIANT_NAME}.mp4"
encode_video(FRAME_DIR, video_path, NUM_FRAMES, VIDEO_FPS)
detector.close()
plotter.close()
print(f"Completed {NUM_FRAMES} frames in {time.perf_counter() - simulation_start:.2f} s.", flush=True)
print(f"Saved video: {video_path}", flush=True)
print(f"Saved surface states: {SURFACE_DIR}", flush=True)
