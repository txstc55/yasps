from pathlib import Path
import json
import os
import sys
import time
import numpy as np
import pyvista as pv
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from yasps import scene, attribute
from helpers import inertia, stable_neo_hookean, affine_matrix_target, affine_translation_target, orthogonality, determinant, joint_pivot, hinge_axis, fixed_joint_rotation, point_point, point_edge, point_triangle, edge_edge, tet_vertex_masses, tet_signed_volumes, surface_vertex_masses, triangle_cells, save_video
from friction_helpers import closest_point_coord_and_tangent_basis_pp, closest_point_coord_and_tangent_basis_pe, closest_point_coord_and_tangent_basis_pt, closest_point_coord_and_tangent_basis_ee, lambda_last_h_pp, lambda_last_h_pe, lambda_last_h_pt, lambda_last_h_ee, friction_energy_pp, friction_energy_pe, friction_energy_pt, friction_energy_ee

##################################################################
## Settings: implicit Euler, a palm-up affine hand, and a rubbery bunny.
## Edit these constants directly; there is no inverse optimization here.
##################################################################
EXAMPLE_DIR = Path(__file__).resolve().parent
os.chdir(EXAMPLE_DIR)
sys.path.insert(0, str(EXAMPLE_DIR.parent / "ccd"))
from ccd import CCD
NUM_FRAMES = 200
DT_VALUE = 0.01
MOTION_TOLERANCE = 1.0e-2
CG_TOLERANCE = 1.0e-4
MAX_LINE_SEARCH_STEPS = 24
GRAVITY_VALUE = 9.81
HAND_MASS = 0.4
BUNNY_DENSITY = 1100.0
BUNNY_YOUNG = 300000.0
BUNNY_POISSON = 0.45
POSE_TRANSLATION_STIFFNESS = 1000.0
WRIST_TRANSLATION_STIFFNESS = 100000.0
ROTATION_STIFFNESS = 1000.0
JOINT_PIVOT_STIFFNESS = 100000.0
JOINT_AXIS_STIFFNESS = 1000.0
DHAT_VALUE = (0.0002) ** 2  # Squared activation distance: 0.2 mm.
KAPPA_VALUE = 1.0e3
FRICTION_RATE = 0.5  # Higher friction for a more secure bunny grasp.
SHOW_GUI = True
VIDEO_FPS = 100
OUTPUT_DIR = EXAMPLE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "surface_frames").mkdir(exist_ok=True)
assets = dict(np.load(EXAMPLE_DIR / "simulation_assets" / "grasp_assets.npz"))
assert float(assets["dt"]) == DT_VALUE and len(assets["target_global"]) >= NUM_FRAMES
rest_global = assets["rest_global"]
inverse_bind = np.linalg.inv(rest_global)
hand_rest = assets["hand_vertices"]
bunny_rest = assets["bunny_vertices"]
bunny_tets = assets["bunny_tets"]
hand_masses = surface_vertex_masses(hand_rest, assets["hand_triangles"], HAND_MASS)
bunny_masses, rest_volumes = tet_vertex_masses(bunny_rest, bunny_tets, BUNNY_DENSITY)
bone_count = len(rest_global)
grouped_to_original = assets["hand_union_to_original"]
original_to_grouped = assets["hand_original_to_union"]
bunny_offset = int(assets["bunny_offset"])
bunny_surface_ids = assets["bunny_surface_indices"]
bunny_to_surface = np.full(len(bunny_rest), -1, dtype=np.int64)
bunny_to_surface[bunny_surface_ids] = np.arange(len(bunny_surface_ids))
bunny_surface_triangles = bunny_to_surface[assets["bunny_surface_triangles"]]
np.testing.assert_array_equal(assets["union_surface_indices"], np.unique(assets["union_triangles"]))
np.testing.assert_array_equal(assets["union_vertices"][assets["hand_triangles_union"]], hand_rest[assets["hand_triangles"]])
np.testing.assert_array_equal(assets["union_vertices"][assets["bunny_triangles_union"]], bunny_rest[assets["bunny_surface_triangles"]])

##################################################################
## Scene constants and the 24 independent global affine bone transforms.
## A maps a bone-local bind point to world space; t is its global origin.
##################################################################
s0 = scene("hand_bunny_grasp")
dt = s0.addConstant("dt"); dt.updateValue(DT_VALUE)
gravity = s0.addConstant("gravity"); gravity.updateValue(GRAVITY_VALUE)
dhat = s0.addConstant("dhat"); dhat.updateValue(DHAT_VALUE)
kappa = s0.addConstant("kappa"); kappa.updateValue(KAPPA_VALUE)
friction_rate = s0.addConstant("friction_rate"); friction_rate.updateValue(FRICTION_RATE)
mu = s0.addConstant("mu"); mu.updateValue(BUNNY_YOUNG / (2.0 * (1.0 + BUNNY_POISSON)))
lam = s0.addConstant("lam"); lam.updateValue(BUNNY_YOUNG * BUNNY_POISSON / ((1.0 + BUNNY_POISSON) * (1.0 - 2.0 * BUNNY_POISSON)))
rotation_weight = s0.addConstant("rotation_weight"); rotation_weight.updateValue(ROTATION_STIFFNESS)
pivot_weight = s0.addConstant("pivot_weight"); pivot_weight.updateValue(JOINT_PIVOT_STIFFNESS)
axis_weight = s0.addConstant("axis_weight"); axis_weight.updateValue(JOINT_AXIS_STIFFNESS)
hand = s0.addMesh("hand")
bones = hand.addPrimitive("affine_bones", numInstances=bone_count)
affine_matrix = bones.addAttribute("affine_matrix", rows=3, cols=3)
affine_translation = bones.addAttribute("affine_translation", rows=3, cols=1)
affine_matrix.updateValue(rest_global[:, :3, :3].ravel())
affine_translation.updateValue(rest_global[:, :3, 3].ravel())
bone_rest_position = bones.addConstant("rest_position", rows=3, cols=1)
bone_rest_position.updateValue(np.zeros(bone_count * 3))
bone_position = bones.addAttribute("position", computed_attribute=affine_matrix * bone_rest_position + affine_translation)
matrix_target = bones.addConstant("matrix_target", rows=3, cols=3)
translation_target = bones.addConstant("translation_target", rows=3, cols=1)
matrix_target.updateValue(rest_global[:, :3, :3].ravel())
translation_target.updateValue(rest_global[:, :3, 3].ravel())
translation_stiffness = np.full(bone_count, POSE_TRANSLATION_STIFFNESS)
translation_stiffness[0] = WRIST_TRANSLATION_STIFFNESS  # The wrist anchors the held hand against gravity.
matrix_stiffness = translation_stiffness * np.maximum(assets["bone_lengths"], 0.01) ** 2
matrix_weight = bones.addConstant("matrix_weight"); matrix_weight.updateValue(matrix_stiffness)
translation_weight = bones.addConstant("translation_weight"); translation_weight.updateValue(translation_stiffness)
matrix_energy = bones.addAttribute("matrix_target_energy", computed_attribute=affine_matrix_target(affine_matrix, matrix_target, matrix_weight, dt))
translation_energy = bones.addAttribute("translation_target_energy", computed_attribute=affine_translation_target(affine_translation, translation_target, translation_weight, dt))
orthogonal_energy = bones.addAttribute("orthogonal_energy", computed_attribute=orthogonality(affine_matrix, rotation_weight, dt))
determinant_energy = bones.addAttribute("determinant_energy", computed_attribute=determinant(affine_matrix, rotation_weight, dt))
s0.addEnergy(matrix_energy, projection_method=-1)
s0.addEnergy(translation_energy, projection_method=-1)
s0.addEnergy(orthogonal_energy, projection_method=1)
s0.addEnergy(determinant_energy, projection_method=1)

##################################################################
## Exact-arity skin groups: no padded influences or discarded weights.
## Each skin point sums w_b (A_b X_bind_local_b + t_b).
## Compute each influence through a one-bone JOIN before joining the k
## contributions. This repeats the same 12-column bone layout for every
## arity, so UNION branches do not mix X/Y/Z Jacobian column supports.
## Physical inertia lives on these skin vertices, not twice on the bones.
## A separate one-to-one JOIN exposes each computed position to inertia:
## its inner Hessian is 3x3, with a 3x(12k) skinning Jacobian for k bones.
##################################################################
skin_primitives = []
skin_positions = []
last_positions = []
velocities = []
for group, count_value in enumerate(assets["group_counts"]):
  count = int(count_value)
  begin, end = assets["group_offsets"][group:group + 2]
  original_ids = grouped_to_original[begin:end]
  bone_ids = assets[f"group_{count}_bones"]
  weights = assets[f"group_{count}_weights"]
  bind_points = np.einsum("vkij,vj->vki", inverse_bind[bone_ids], np.column_stack([hand_rest[original_ids], np.ones(len(original_ids))]))[:, :, :3]
  influences = hand.addPrimitive(f"influences_{count}_bones", numInstances=len(original_ids) * count)
  influence_connection = influences.addConnectivity("to_bones", bones, bone_ids.ravel(), 1)
  matrices = influences.addAttribute("matrices", through=influence_connection, source=affine_matrix).resize(3, 3)
  translations = influences.addAttribute("translations", through=influence_connection, source=affine_translation).resize(3, 1)
  bind = influences.addConstant("bind_points", rows=3, cols=1); bind.updateValue(bind_points.ravel())
  influence_position = influences.addAttribute("position", computed_attribute=matrices * bind + translations)
  vertices = hand.addPrimitive(f"skin_{count}_bones", numInstances=len(original_ids))
  connection = vertices.addConnectivity("to_influences", influences, np.arange(len(original_ids) * count, dtype=np.uint32), count)
  contributions = vertices.addAttribute("contributions", through=connection, source=influence_position)
  weight = vertices.addConstant("weights", rows=count, cols=1); weight.updateValue(weights.ravel())
  rest = vertices.addConstant("rest_position", rows=3, cols=1); rest.updateValue(hand_rest[original_ids].ravel())
  expression = sum(weight[j, 0] * contributions.row(j).transpose() for j in range(count))
  position = vertices.addAttribute("position", computed_attribute=expression)
  inertia_vertices = hand.addPrimitive(f"inertia_{count}_bones", numInstances=len(original_ids))
  inertia_connection = inertia_vertices.addConnectivity("to_skin", vertices, np.arange(len(original_ids), dtype=np.uint32), 1)
  joined_position = inertia_vertices.addAttribute("joined_position", through=inertia_connection, source=position)
  inertia_position = inertia_vertices.addAttribute("position", computed_attribute=joined_position.resize(3, 1))
  last = inertia_vertices.addConstant("last_position", rows=3, cols=1); last.updateValue(hand_rest[original_ids].ravel())
  velocity = inertia_vertices.addConstant("velocity", rows=3, cols=1); velocity.updateValue(np.zeros(len(original_ids) * 3))
  mass = inertia_vertices.addConstant("mass"); mass.updateValue(hand_masses[original_ids])
  energy = inertia_vertices.addAttribute("inertia_energy", computed_attribute=inertia(last, velocity, dt, inertia_position, mass, gravity))
  s0.addEnergy(energy, projection_method=-1, separate_hessian_jacobian=True)
  skin_primitives.append(inertia_vertices)
  skin_positions.append(position)
  last_positions.append(last)
  velocities.append(velocity)

##################################################################
## Joint centers stay connected; nine anatomical joints are hinges.
## The axis residual compares GLOBAL axes computed from different local
## bind frames. With proper rotations it leaves only twist about that axis.
##################################################################
joints = hand.addPrimitive("joints", numInstances=len(assets["joint_pairs"]))
joint_connection = joints.addConnectivity("to_bones", bones, assets["joint_pairs"], 2)
joint_matrices = joints.addAttribute("matrices", through=joint_connection, source=affine_matrix)
joint_translations = joints.addAttribute("translations", through=joint_connection, source=affine_translation)
parent_pivot = joints.addConstant("parent_pivot", rows=3, cols=1); parent_pivot.updateValue(assets["joint_rest_relative_matrices"][:, :3, 3].ravel())
child_pivot = joints.addConstant("child_pivot", rows=3, cols=1); child_pivot.updateValue(np.zeros(len(assets["joint_pairs"]) * 3))
pivot_energy = joints.addAttribute("pivot_energy", computed_attribute=joint_pivot(joint_matrices, joint_translations, parent_pivot, child_pivot, pivot_weight, dt))
s0.addEnergy(pivot_energy, projection_method=-1, separate_hessian_jacobian=True)
hinges = hand.addPrimitive("hinges", numInstances=len(assets["hinge_pairs"]))
hinge_connection = hinges.addConnectivity("to_bones", bones, assets["hinge_pairs"], 2)
hinge_matrices = hinges.addAttribute("matrices", through=hinge_connection, source=affine_matrix)
parent_axis = hinges.addConstant("parent_axis", rows=3, cols=1); parent_axis.updateValue(assets["hinge_axes_parent_local"].ravel())
child_axis = hinges.addConstant("child_axis", rows=3, cols=1); child_axis.updateValue(assets["hinge_axes_child_local"].ravel())
hinge_energy = hinges.addAttribute("hinge_energy", computed_attribute=hinge_axis(hinge_matrices, parent_axis, child_axis, axis_weight, dt))
s0.addEnergy(hinge_energy, projection_method=-1, separate_hessian_jacobian=True)
fixed_ids = np.flatnonzero(assets["joint_types"] == "fixed_marker")
markers = hand.addPrimitive("fixed_markers", numInstances=len(fixed_ids))
marker_connection = markers.addConnectivity("to_bones", bones, assets["joint_pairs"][fixed_ids], 2)
marker_matrices = markers.addAttribute("matrices", through=marker_connection, source=affine_matrix)
marker_relative = markers.addConstant("rest_relative", rows=3, cols=3); marker_relative.updateValue(assets["joint_rest_relative_matrices"][fixed_ids, :3, :3].ravel())
marker_energy = markers.addAttribute("rotation_energy", computed_attribute=fixed_joint_rotation(marker_matrices, marker_relative, axis_weight, dt))
s0.addEnergy(marker_energy, projection_method=-1, separate_hessian_jacobian=True)

##################################################################
## Volumetric bunny, with density-lumped masses and stable neo-Hookean FEM.
##################################################################
bunny = s0.addMesh("bunny")
bunny_vertices = bunny.addPrimitive("vertices", numInstances=len(bunny_rest))
bunny_position = bunny_vertices.addAttribute("position", rows=3, cols=1); bunny_position.updateValue(bunny_rest.ravel())
bunny_rest_position = bunny_vertices.addConstant("rest_position", rows=3, cols=1); bunny_rest_position.updateValue(bunny_rest.ravel())
bunny_last = bunny_vertices.addConstant("last_position", rows=3, cols=1); bunny_last.updateValue(bunny_rest.ravel())
bunny_velocity = bunny_vertices.addConstant("velocity", rows=3, cols=1); bunny_velocity.updateValue(np.zeros(bunny_rest.size))
bunny_mass = bunny_vertices.addConstant("mass"); bunny_mass.updateValue(bunny_masses)
bunny_inertia = bunny_vertices.addAttribute("inertia_energy", computed_attribute=inertia(bunny_last, bunny_velocity, dt, bunny_position, bunny_mass, gravity))
tets = bunny.addPrimitive("tets", numInstances=len(bunny_tets))
tet_connection = tets.addConnectivity("to_vertices", bunny_vertices, bunny_tets, 4)
tet_rest = tets.addAttribute("rest_positions", through=tet_connection, source=bunny_rest_position)
tet_current = tets.addAttribute("positions", through=tet_connection, source=bunny_position)
elastic_energy = tets.addAttribute("elastic_energy", computed_attribute=stable_neo_hookean(tet_rest, tet_current, mu, lam, dt))
s0.addEnergy(bunny_inertia, projection_method=-1)
s0.addEnergy(elastic_energy, projection_method=1)
vertex_positions = skin_positions + [bunny_position]
last_positions.append(bunny_last)
velocities.append(bunny_velocity)

##################################################################
## Union uses the one-to-one position JOIN primitives for skin_1, ...,
## skin_7, followed by ALL bunny vertices; vertex numbering is unchanged.
## CCD only receives the remapped surface vertex IDs and surface topology.
## Mesh ID zero enables self-contact and contact between hand and bunny.
##################################################################
collisions = s0.addMesh("collisions")
union = collisions.addPrimitiveUnion("vertices", skin_primitives + [bunny_vertices])
union_position = union.addAttribute("position")
friction_last_position = union.addConstant("friction_last_position", rows=3, cols=1)
friction_last_position.updateValue(assets["union_vertices"].ravel())
collision_primitives = []
collision_connections = []
for name, arity, barrier in [("pp", 2, point_point), ("pe", 3, point_edge), ("pt", 4, point_triangle), ("ee", 4, edge_edge)]:
  primitive = collisions.addPrimitive(name, numInstances=0, isDynamic=True)
  connection = primitive.addConnectivity("to_vertices", union, [], arity)
  positions = primitive.addAttribute("positions", through=connection, source=union_position)
  energy = primitive.addAttribute("energy", computed_attribute=barrier(positions, dhat, kappa))
  s0.addEnergy(energy, dynamic_instances=True, projection_method=2, separate_hessian_jacobian=True)
  collision_primitives.append(primitive)
  collision_connections.append(connection)

##################################################################
## Lagged friction uses the previous converged union positions. Its own
## contact set is frozen throughout each frame, including line search.
## Current positions still differentiate through skinning into both the
## affine bones and bunny DOFs; there is no one-sided target restriction.
##################################################################
friction_primitives = []
friction_connections = []
for name, arity, closest, normal_force, friction in [("pp", 2, closest_point_coord_and_tangent_basis_pp, lambda_last_h_pp, friction_energy_pp), ("pe", 3, closest_point_coord_and_tangent_basis_pe, lambda_last_h_pe, friction_energy_pe), ("pt", 4, closest_point_coord_and_tangent_basis_pt, lambda_last_h_pt, friction_energy_pt), ("ee", 4, closest_point_coord_and_tangent_basis_ee, lambda_last_h_ee, friction_energy_ee)]:
  primitive = collisions.addPrimitive(f"{name}_friction", numInstances=0, isDynamic=True)
  connection = primitive.addConnectivity("to_vertices", union, [], arity)
  positions = primitive.addAttribute("positions", through=connection, source=union_position)
  old_positions = primitive.addAttribute("last_positions", through=connection, source=friction_last_position)
  coord, tangent_basis = closest(old_positions)
  primitive.addAttribute("coord", computed_attribute=coord)
  primitive.addAttribute("tangent_basis", computed_attribute=tangent_basis)
  normal_load = normal_force(old_positions, coord, dhat, kappa)
  primitive.addAttribute("lambda_last_h", computed_attribute=normal_load)
  energy = primitive.addAttribute("friction_energy", computed_attribute=friction(positions, old_positions, dhat, dt, friction_rate, coord, tangent_basis.row(0), tangent_basis.row(1), normal_load))
  s0.addEnergy(energy, dynamic_instances=True, projection_method=2, separate_hessian_jacobian=True)
  friction_primitives.append(primitive)
  friction_connections.append(connection)

##################################################################
## Verify actual GPU skinning before differentiation and contact assembly.
## This checks every arity and the same union order consumed by CCD.
##################################################################
maximum_replay_error = 0.0
for verification_frame in [-1, 0, 50, 100, 150, 199]:
  pose = rest_global if verification_frame == -1 else assets["target_global"][verification_frame]
  reference = hand_rest if verification_frame == -1 else assets["target_surface_positions"][verification_frame]
  affine_matrix.updateValue(pose[:, :3, :3].ravel())
  affine_translation.updateValue(pose[:, :3, 3].ravel())
  gpu_positions = union_position.compute().value.get().reshape(-1, 3)
  error = float(np.max(np.abs(gpu_positions[:bunny_offset][original_to_grouped] - reference)))
  maximum_replay_error = max(maximum_replay_error, error)
  np.testing.assert_allclose(gpu_positions[:bunny_offset][original_to_grouped], reference, rtol=0, atol=1e-12)
  np.testing.assert_allclose(gpu_positions[bunny_offset:], bunny_rest, rtol=0, atol=1e-12)
  for check in [pivot_energy, hinge_energy, marker_energy, orthogonal_energy, determinant_energy]:
    assert np.abs(check.compute().value.get()).max() < 1e-20
affine_matrix.updateValue(rest_global[:, :3, :3].ravel())
affine_translation.updateValue(rest_global[:, :3, 3].ravel())
print(f"GPU grouped-skin/union replay maximum error: {maximum_replay_error:.3e} m", flush=True)
gpu_free_at_start, gpu_total = cuda.mem_get_info()
setup_start = time.perf_counter()
targets = [bunny_position, affine_matrix, affine_translation]
s0.addMinimizeTarget(targets)
ccd = CCD(len(assets["union_surface_indices"]), len(assets["union_vertices"]), max_cd_pairs=2000000, max_ccd_pairs=20000000, mesh_indices=np.zeros(len(assets["union_vertices"]), dtype=np.uint32), print_timings=False)
initial_positions = union_position.compute().value.copy()
ccd.init_faces(initial_positions, gpuarray.to_gpu(assets["union_triangles"].ravel()), gpuarray.to_gpu(assets["union_surface_indices"]), len(assets["union_triangles"]))
ccd.init_edges(initial_positions, initial_positions, gpuarray.to_gpu(assets["union_edges"].ravel()), len(assets["union_edges"]))
print(f"Ready: hand={bunny_offset} vertices, bones={bone_count}, bunny={len(bunny_rest)} vertices/{len(bunny_tets)} tets, DOF={3 * len(bunny_rest) + 12 * bone_count}, bunny mass={bunny_masses.sum():.6f} kg", flush=True)

##################################################################
## Fixed oblique camera, Y up; keep the whole held hand in view.
##################################################################
hand_poly = pv.PolyData(hand_rest.copy(), triangle_cells(assets["hand_triangles"]))
bunny_poly = pv.PolyData(bunny_rest[bunny_surface_ids].copy(), triangle_cells(bunny_surface_triangles))
plotter = pv.Plotter(window_size=(1600, 1000), off_screen=not SHOW_GUI)
plotter.set_background("#eef1f5")
plotter.add_mesh(hand_poly, color="#d6aa82", smooth_shading=True, specular=0.2)
plotter.add_mesh(bunny_poly, color="#438fca", smooth_shading=True, specular=0.25)
plotter.camera_position = [(-0.17, 0.24, -0.30), (0.075, 0.025, 0.0), (0, 1, 0)]
plotter.camera.parallel_projection = True
plotter.camera.parallel_scale = 0.12
plotter.camera.clipping_range = (0.01, 2.0)
plotter.add_text("Palm-up affine hand + rubbery FEM bunny", position="upper_left", font_size=14, color="#263449")
plotter.show(interactive_update=True, auto_close=False)
plotter.screenshot(str(OUTPUT_DIR / "rest_pose.jpg"))
np.savez_compressed(OUTPUT_DIR / "surface_rest.npz", hand_positions=hand_rest, hand_triangles=assets["hand_triangles"], bunny_positions=bunny_rest[bunny_surface_ids], bunny_triangles=bunny_surface_triangles)

##################################################################
## Forward frames. Targets are forces, not hard position assignments:
## contact therefore acts back on the bones and can stop the closing hand.
##################################################################
frame_statistics = []
saved_global = []
simulation_start = time.perf_counter()
for frame in range(NUM_FRAMES):
  frame_start = time.perf_counter()
  desired = assets["target_global"][frame]
  matrix_target.updateValue(desired[:, :3, :3].ravel())
  translation_target.updateValue(desired[:, :3, 3].ravel())
  for last, position in zip(last_positions, vertex_positions):
    last.updateValue(position.compute().value, deepCopy=True)
  # Rebuild friction from this exact frame-start state, not a stale CCD
  # trial. Connectivity copies own their buffers, so later CD cannot change it.
  friction_last_position.updateValue(union_position.compute().value, deepCopy=True)
  ccd.cd(friction_last_position.value, DHAT_VALUE)
  friction_counts = list(map(int, ccd.separated_counts))
  for primitive, connection, arity, pairs, count in zip(friction_primitives, friction_connections, [2, 3, 4, 4], [ccd.pp, ccd.pe, ccd.pt, ccd.ee], friction_counts):
    primitive.updateNumInstances(count)
    if count:
      connection.updateConnectivity(pairs[:arity * count])
  print(f"FRICTION_SET frame={frame:03d} contacts={friction_counts} coefficient={FRICTION_RATE}", flush=True)
  newton_iteration = 0
  cg_iterations = 0
  line_search_reductions = 0
  solve_seconds = 0.0
  minimum_ccd_step = 1.0
  while True:
    # Refresh the discrete contact set at the exact current Newton iterate.
    current_union = union_position.compute().value.copy()
    ccd.cd(current_union, DHAT_VALUE)
    for primitive, connection, arity, pairs, count in zip(collision_primitives, collision_connections, [2, 3, 4, 4], [ccd.pp, ccd.pe, ccd.pt, ccd.ee], ccd.separated_counts):
      primitive.updateNumInstances(count)
      if count:
        connection.updateConnectivity(pairs[:arity * count])
    energy_before = s0.computeTotalEnergy()
    solve_start = time.perf_counter()
    exit_code = s0.minimizer.computeHessianAndGradient(tolerance=CG_TOLERANCE)
    solve_seconds += time.perf_counter() - solve_start
    if exit_code < 0:
      print(f"SOLVER_FAILURE {json.dumps(s0.minimizer.linearSolver.statistics)}", flush=True)
      raise RuntimeError(f"GPU solver failed at frame {frame}, Newton {newton_iteration}: {exit_code}; no invalid step was accepted.")
    directions = s0.minimizer.solutionSegments
    cg_iterations += int(s0.minimizer.linearSolver.statistics.get("iterations", 0))
    saved_values = [target.compute().value.copy() for target in targets]
    for target, original, direction in zip(targets, saved_values, directions):
      target.updateValue(original - direction, deepCopy=True)
    displacement = current_union - union_position.compute().value
    maximum_displacement = float(gpuarray.max(abs(displacement)).get())
    motion = maximum_displacement / DT_VALUE
    print(f"NEWTON_DIRECTION frame={frame:03d} iteration={newton_iteration:03d} max_displacement={maximum_displacement:.6e} m", flush=True)
    # All skinning is affine in these DOFs, so this swept segment is exact.
    # A very long Newton proposal can overflow broad-phase storage. Shorten
    # the trial interval and rebuild CCD; never bypass collision checking.
    sweep_alpha = 1.0
    for sweep_attempt in range(MAX_LINE_SEARCH_STEPS):
      try:
        ccd.ccd(current_union, DHAT_VALUE, displacement, sweep_alpha)
        break
      except OverflowError as error:
        if "broad-phase candidate capacity" not in str(error):
          raise
        sweep_alpha *= 0.5
        print(f"CCD_SWEEP_RETRY alpha={sweep_alpha:.6e}", flush=True)
    else:
      for target, original in zip(targets, saved_values):
        target.updateValue(original, deepCopy=True)
      raise RuntimeError("CCD candidate storage still overflowed after shortening the trial interval.")
    step_size = float(ccd.compute_largest_step_size(0.5, current_union, displacement))
    minimum_ccd_step = min(minimum_ccd_step, step_size)
    for line_search in range(MAX_LINE_SEARCH_STEPS):
      for target, original, direction in zip(targets, saved_values, directions):
        target.updateValue(original - step_size * direction, deepCopy=True)
      trial_union = union_position.compute().value
      ccd.cd(trial_union, DHAT_VALUE)
      for primitive, connection, arity, pairs, count in zip(collision_primitives, collision_connections, [2, 3, 4, 4], [ccd.pp, ccd.pe, ccd.pt, ccd.ee], ccd.separated_counts):
        primitive.updateNumInstances(count)
        if count:
          connection.updateConnectivity(pairs[:arity * count])
      energy_after = s0.computeTotalEnergy()
      if np.isfinite(energy_after) and energy_after <= energy_before + 1e-14 * max(1.0, abs(energy_before)):
        break
      step_size *= 0.5
    else:
      for target, original in zip(targets, saved_values):
        target.updateValue(original, deepCopy=True)
      raise RuntimeError(f"Line search failed at frame {frame}, Newton {newton_iteration}; restored the previous safe state.")
    newton_iteration += 1
    line_search_reductions += line_search
    print(f"frame={frame:03d} newton={newton_iteration:03d} motion={motion:.6e} step={step_size:.6e} contacts={ccd.separated_counts} energy={energy_after:.9e}", flush=True)
    # Use the full Newton movement, not the accepted step-scaled movement.
    # Always perform CCD/line search before testing convergence.
    if motion < MOTION_TOLERANCE:
      break
  for velocity, last, position in zip(velocities, last_positions, vertex_positions):
    velocity.updateValue((position.compute().value - last.value) / DT_VALUE, deepCopy=True)

  ################################################################
  ## Save the converged surface, bone transforms, physical diagnostics,
  ## and one JPG. Interior tet vertices are not exported as surface points.
  ################################################################
  current = union_position.compute().value.get().reshape(-1, 3)
  hand_current = current[:bunny_offset][original_to_grouped]
  bunny_current = current[bunny_offset:]
  global_pose = np.repeat(np.eye(4)[None], bone_count, axis=0)
  global_pose[:, :3, :3] = affine_matrix.value.get().reshape(-1, 3, 3)
  global_pose[:, :3, 3] = affine_translation.value.get().reshape(-1, 3)
  saved_global.append(global_pose)
  matrices_cpu = global_pose[:, :3, :3]
  parent, child = assets["joint_pairs"].T
  pivot_residual = np.einsum("nij,nj->ni", matrices_cpu[parent], assets["joint_rest_relative_matrices"][:, :3, 3]) + global_pose[parent, :3, 3] - global_pose[child, :3, 3]
  hp, hc = assets["hinge_pairs"].T
  axis_residual = np.einsum("nij,nj->ni", matrices_cpu[hp], assets["hinge_axes_parent_local"]) - np.einsum("nij,nj->ni", matrices_cpu[hc], assets["hinge_axes_child_local"])
  jacobians = tet_signed_volumes(bunny_current, bunny_tets) / rest_volumes
  gpu_free, _ = cuda.mem_get_info()
  stats = {"frame": frame, "time": (frame + 1) * DT_VALUE, "newton_iterations": newton_iteration, "cg_iterations": cg_iterations, "hessian_and_solve_seconds": solve_seconds, "frame_seconds": time.perf_counter() - frame_start, "line_search_reductions": line_search_reductions, "minimum_ccd_step": minimum_ccd_step, "contacts": list(map(int, ccd.separated_counts)), "maximum_motion": motion, "minimum_bunny_detF": float(jacobians.min()), "minimum_bone_determinant": float(np.linalg.det(matrices_cpu).min()), "maximum_rotation_error": float(np.abs(matrices_cpu.transpose(0, 2, 1) @ matrices_cpu - np.eye(3)).max()), "maximum_joint_gap_m": float(np.linalg.norm(pivot_residual, axis=1).max()), "maximum_hinge_axis_error": float(np.linalg.norm(axis_residual, axis=1).max()), "gpu_memory_gib": (gpu_total - gpu_free) / 2**30, "gpu_memory_added_gib": (gpu_free_at_start - gpu_free) / 2**30}
  frame_statistics.append(stats)
  stats["friction_contacts"] = friction_counts
  stats["friction_coefficient"] = FRICTION_RATE
  np.savez_compressed(OUTPUT_DIR / "surface_frames" / f"frame_{frame:04d}.npz", frame=frame, time=stats["time"], hand_positions=hand_current, hand_triangles=assets["hand_triangles"], bunny_positions=bunny_current[bunny_surface_ids], bunny_triangles=bunny_surface_triangles, bone_global=global_pose, bone_target=desired)
  hand_poly.points = hand_current
  bunny_poly.points = bunny_current[bunny_surface_ids]
  # Recompute shading normals after deformation; do not reuse rest normals.
  hand_poly.point_data.remove("Normals")
  bunny_poly.point_data.remove("Normals")
  hand_poly.compute_normals(cell_normals=False, inplace=True)
  bunny_poly.compute_normals(cell_normals=False, inplace=True)
  plotter.add_text(f"Frame {frame + 1:03d}/{NUM_FRAMES}  |  t = {stats['time']:.2f} s", position="lower_left", font_size=13, color="#263449", name="status")
  plotter.update()
  plotter.screenshot(str(OUTPUT_DIR / f"frame_{frame:04d}.jpg"))
  (OUTPUT_DIR / "statistics.json").write_text(json.dumps({"dt": DT_VALUE, "frames": frame_statistics, "gpu_replay_max_error_m": maximum_replay_error, "bunny_mass_kg": float(bunny_masses.sum())}, indent=2) + "\n")
  print(f"FRAME_DONE {json.dumps(stats)}", flush=True)

##################################################################
## Full affine trajectory, final volume state, and the 100 fps video.
##################################################################
np.savez_compressed(OUTPUT_DIR / "bone_trajectory.npz", global_affine=np.asarray(saved_global), target_global=assets["target_global"][:NUM_FRAMES], rest_global=rest_global, bone_names=assets["bone_names"], parents=assets["parents"], times=DT_VALUE * np.arange(1, NUM_FRAMES + 1), dt=DT_VALUE)
np.savez_compressed(OUTPUT_DIR / "final_state.npz", bone_global=saved_global[-1], bunny_positions=bunny_current, bunny_tets=bunny_tets, bunny_rest=bunny_rest, bunny_velocity=bunny_velocity.value.get().reshape(-1, 3))
video_path = save_video(OUTPUT_DIR, NUM_FRAMES, VIDEO_FPS)
plotter.close()
print(f"Completed {NUM_FRAMES} forward frames in {time.perf_counter() - simulation_start:.2f} s. Video: {video_path}", flush=True)
