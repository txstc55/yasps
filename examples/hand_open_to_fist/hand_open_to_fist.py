from pathlib import Path
import json
import os
import sys
import time
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation
from yasps import scene
from helpers import rotation_matrices, build_joint_constraints, constrained_joint_rotations, hinge_residuals, forward_kinematics, skin_vertices, build_yasps_skinning, update_rotations, make_collision_detector, safe_segment, self_intersections, save_video

##################################################################
## Settings: kinematic forward motion, no inverse or dynamic solve.
##################################################################
EXAMPLE_DIR = Path(__file__).resolve().parent
os.chdir(EXAMPLE_DIR)
sys.path.insert(0, str(EXAMPLE_DIR.parent / "ccd"))
import ccd
NUM_FRAMES = 200
DT_VALUE = 0.01
CLEARANCE = 0.00005  # 0.05 mm between nonincident surface primitives.
SHOW_GUI = True
KEEP_FINAL_WINDOW_OPEN = False  # Close after export so the grasp simulation can run next.
VIDEO_FPS = 100
MOTION_START = 0.10
MOTION_END = 1.80
RELAXED_HINGES = []  # Explicit pose-specific exceptions; pivots remain joined.
OUTPUT_DIR = EXAMPLE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
ASSET_PATH = EXAMPLE_DIR / "simulation_assets" / "hand_bind.npz"
asset = dict(np.load(ASSET_PATH))
names = asset["bone_names"].tolist()
indices = {name: i for i, name in enumerate(names)}
rest = asset["vertices"]
triangles = asset["triangles"]
weights = asset["weights"]
inverse_rest = np.linalg.inv(asset["rest_global"])

##################################################################
## Fixed-axis joints and their parent/child frames for later energies.
## PIP/DIP and thumb IP are hinges; MCP/CMC joints are not locked to Z.
##################################################################
constraints = build_joint_constraints(asset, RELAXED_HINGES)
# Keep the trajectory self-contained while retaining its existing bind keys.
constraint_fields = {key: value for key, value in constraints.items() if key not in ("bone_names", "parents", "rest_global_matrices")}
hinge_children = constraints["hinge_pairs"][:, 1]

##################################################################
## One coordinated closure: every joint follows the same progress.
## The thumb curls directly into opposition, without a lift waypoint.
##################################################################
target_angles = np.zeros((len(names), 3))
for finger, spread in [("index", 0), ("middle", 0), ("ring", -5), ("pinky", -8)]:
  group = np.asarray([indices[f"b_l_{finger}{j}"] for j in (1, 2, 3)])
  target_angles[group, 2] = np.deg2rad([82.0, 74.0 if finger == "ring" else 75.0, 70.0])
  target_angles[group[0], 1] = np.deg2rad(spread)
thumb_group = np.asarray([indices[f"b_l_thumb{j}"] for j in range(4)])
# Bring the finger pads closer to the palm while wrapping the thumb outside
# them. Its base also moves, so the tighter fingers clear the thenar surface.
# All joints still share the same smooth progress; none waits at contact.
target_angles[thumb_group] = np.deg2rad([[-0.126255, 55.0, 4.622035], [0.0, 8.813156, 65.0], [0.0, 0.0, 18.549689], [0.0, 0.0, 70.560578]])
target_rotations = rotation_matrices(target_angles)
target_rotation_vectors = Rotation.from_matrix(target_rotations).as_rotvec()
hinge_target_angles = np.einsum("hi,hi->h", target_rotation_vectors[hinge_children], constraints["hinge_axes_child_local"])
# Validate target axes before building the skinning graph or generating frames.
initial_rotations = constrained_joint_rotations(target_rotation_vectors, 0.0, constraints)
angles = np.zeros_like(target_angles)
positions = rest.copy()

##################################################################
## Native YASPS parent transforms and weighted surface interpolation.
##################################################################
model = scene("hand_open_to_fist")
rotation_updates, surface_position, bone_global_attribute, bone_order = build_yasps_skinning(model, asset)
update_rotations(rotation_updates, initial_rotations)
np.testing.assert_allclose(surface_position.compute().value.get().reshape(-1, 3), rest, atol=1e-12)
detector, gpu_position, gpu_direction = make_collision_detector(rest, triangles, ccd)
assert len(self_intersections(rest, triangles)) == 0
assert sum(detector.cd(gpu_position, CLEARANCE * CLEARANCE)) == 0, "Bind surface already violates the requested clearance."

##################################################################
## Fixed oblique palm camera: show the whole hand throughout closure.
##################################################################
surface = pv.PolyData(rest.copy(), np.column_stack([np.full(len(triangles), 3), triangles]))
plotter = pv.Plotter(window_size=(1280, 1000), off_screen=not SHOW_GUI, title="Hand: open to fist")
plotter.set_background("#f5f5f2")
plotter.add_mesh(surface, color="#d5a578", smooth_shading=True, ambient=0.35, diffuse=0.65, specular=0.2)
camera_center = np.array([0.080, 0.020, -0.012])
plotter.camera_position = [camera_center + np.array([0.035, 0.32, -0.22]), camera_center, (1, 0, 0)]
plotter.enable_parallel_projection()
plotter.camera.parallel_scale = 0.135
if SHOW_GUI:
  plotter.show(interactive_update=True, auto_close=False)

##################################################################
## Forward loop: synchronized joint motion, then certify and record.
## Candidate FK is CPU-side; the accepted surface is computed by YASPS.
##################################################################
times = (np.arange(NUM_FRAMES) + 1) * DT_VALUE
local_rotations = np.empty((NUM_FRAMES, len(names), 3, 3))
global_matrices = np.empty((NUM_FRAMES, len(names), 4, 4))
skinning_matrices = np.empty_like(global_matrices)
surface_positions = np.empty((NUM_FRAMES, len(rest), 3))
angles_history = np.empty((NUM_FRAMES, len(names), 3))
progress_history = np.empty(NUM_FRAMES)
checks = []
started = time.perf_counter()
for frame, t in enumerate(times):
  previous_positions = positions.copy()
  u = np.clip((t - MOTION_START) / (MOTION_END - MOTION_START), 0.0, 1.0)
  blend = u * u * u * (10.0 + u * (-15.0 + 6.0 * u))
  # Relative-to-bind rotations take the shortest arc to the final fist. One
  # scalar controls EVERY joint, including the thumb's parent and tip, so no
  # joint can reverse, pause at a waypoint, or finish on a different clock.
  rotations = constrained_joint_rotations(target_rotation_vectors, blend, constraints)
  angles = Rotation.from_matrix(rotations).as_euler("xyz")
  current_global = forward_kinematics(asset["rest_local"], asset["parents"], rotations)
  positions = skin_vertices(rest, weights, current_global, inverse_rest)
  # This fixed trajectory must pass collision checks as a whole. Do not slow
  # only the thumb at contact and silently reintroduce out-of-sync motion.
  safe, step, nearby = safe_segment(detector, gpu_position, gpu_direction, previous_positions, positions, CLEARANCE)
  assert safe, f"The synchronized path is unsafe at frame {frame}: CCD step={step}, near pairs={nearby}; revise the pose rather than desynchronizing the joints."
  intersections = self_intersections(positions, triangles)
  assert not len(intersections), f"Frame {frame} has intersecting triangles."
  update_rotations(rotation_updates, rotations)
  computed = surface_position.compute().value.get().reshape(-1, 3)
  np.testing.assert_allclose(computed, positions, rtol=1e-10, atol=1e-12)
  current_global = bone_global_attribute.compute().value.get().reshape(-1, 4, 4)[np.argsort(bone_order)]
  # Check the generated GLOBAL matrices too: a parent-axis mixup can pass a
  # child-local angle check but give the wrong constraint for an affine solve.
  pivot_error, axis_error = hinge_residuals(current_global, constraints)
  np.testing.assert_allclose(pivot_error, 0.0, rtol=0.0, atol=1e-12)
  np.testing.assert_allclose(axis_error, 0.0, rtol=0.0, atol=1e-12)
  # A relaxed hinge is still attached to its parent. Check every joint pivot,
  # including the unweighted thumb opposition control, not only hinge pairs.
  parent_ids, child_ids = constraints["joint_pairs"].T
  rest_offsets = constraints["joint_rest_relative_matrices"][:, :3, 3]
  all_pivot_error = np.einsum("nij,nj->ni", current_global[parent_ids, :3, :3], rest_offsets) + current_global[parent_ids, :3, 3] - current_global[child_ids, :3, 3]
  np.testing.assert_allclose(all_pivot_error, 0.0, rtol=0.0, atol=1e-12)
  local_rotations[frame] = rotations
  global_matrices[frame] = current_global
  skinning_matrices[frame] = current_global @ inverse_rest
  angles_history[frame] = angles
  progress_history[frame] = blend
  surface_positions[frame] = computed
  checks.append({"frame": frame, "time": float(t), "intersections": len(intersections), "near_pairs": nearby, "ccd_step": step, "closing_progress": float(blend), "max_hinge_pivot_error_m": float(np.max(np.linalg.norm(pivot_error, axis=1), initial=0.0)), "max_hinge_axis_error": float(np.max(np.linalg.norm(axis_error, axis=1), initial=0.0)), "max_all_joint_pivot_error_m": float(np.max(np.abs(all_pivot_error)))})
  surface.points = computed
  plotter.add_text(f"Open to fist | frame {frame + 1}/{NUM_FRAMES} | {t:.2f} s", position="upper_left", font_size=12, color="black", name="status")
  if SHOW_GUI:
    plotter.update()
  plotter.screenshot(str(OUTPUT_DIR / f"frame_{frame:04d}.jpg"))
  print(f"FRAME {frame+1:03d}/{NUM_FRAMES} intersections=0 near_pairs={nearby} CCD={step:.6f} progress={blend:.6f}", flush=True)

##################################################################
## Both transform conventions are saved, so replay is unambiguous.
##################################################################
# Collision-free alone is not enough: do not report a stalled partial closure
# as success. Every digit must reach its prescribed full-fist rotation.
final_target_error = float(np.max(np.abs(rotation_matrices(angles) - rotation_matrices(target_angles))))
assert final_target_error < 1e-5, f"The hand stopped short of the fist target: rotation error {final_target_error}."
np.savez_compressed(OUTPUT_DIR / "hand_trajectory.npz", times=times, dt=DT_VALUE, bone_names=asset["bone_names"], parents=asset["parents"], rest_vertices=rest, triangles=triangles, weights=weights, rest_global_matrices=asset["rest_global"], rest_local_matrices=asset["rest_local"], local_rotations=local_rotations, global_affine_matrices=global_matrices, skinning_affine_matrices=skinning_matrices, surface_positions=surface_positions, local_angles_xyz=angles_history, target_local_rotations=target_rotations, closing_progress=progress_history, motion_start=MOTION_START, motion_end=MOTION_END, hinge_angles=progress_history[:, None] * hinge_target_angles, bone_lengths=asset["bone_lengths"], clearance=CLEARANCE, camera_position=np.asarray(list(plotter.camera_position), dtype=np.float64), camera_parallel_scale=plotter.camera.parallel_scale, **constraint_fields)
(OUTPUT_DIR / "collision_checks.json").write_text(json.dumps(checks, indent=2))
print("Final local angles (degrees):", dict(zip(names, np.rad2deg(angles).tolist())))
print(f"Finished {NUM_FRAMES} frames in {time.perf_counter() - started:.2f}s. Saved {save_video(OUTPUT_DIR, NUM_FRAMES, VIDEO_FPS)}")
detector.close()
if SHOW_GUI and KEEP_FINAL_WINDOW_OPEN:
  plotter.show(auto_close=True)
else:
  plotter.close()
