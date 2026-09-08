"""Rig kinematics, skinning, collision checks, and video export for this example."""
import numpy as np
from scipy.spatial import cKDTree


def build_joint_constraints(asset, relaxed_hinges=()):
  # Anatomical joint classes are assigned from the rig hierarchy, NOT inferred
  # from a single animation (a multi-axis MCP can happen to move about Z only).
  names = asset["bone_names"].tolist()
  parents = asset["parents"]
  definitions = {}
  for finger in ["index", "middle", "ring", "pinky"]:
    base = "b_l_pinky0" if finger == "pinky" else "b_l_wrist"
    definitions[f"b_l_{finger}1"] = (base, "multi_axis", f"{finger}_MCP")
    definitions[f"b_l_{finger}2"] = (f"b_l_{finger}1", "hinge", f"{finger}_PIP")
    definitions[f"b_l_{finger}3"] = (f"b_l_{finger}2", "hinge", f"{finger}_DIP")
    definitions[f"b_l_{finger}_null"] = (f"b_l_{finger}3", "fixed_marker", f"{finger}_tip_marker")
  definitions.update({"b_l_thumb0": ("b_l_wrist", "multi_axis", "thumb_carpal_rig"), "b_l_thumb1": ("b_l_thumb0", "multi_axis", "thumb_CMC"), "b_l_thumb2": ("b_l_thumb1", "multi_axis", "thumb_MCP"), "b_l_thumb3": ("b_l_thumb2", "hinge", "thumb_IP"), "b_l_thumb_null": ("b_l_thumb3", "fixed_marker", "thumb_tip_marker"), "b_l_pinky0": ("b_l_wrist", "multi_axis", "pinky_CMC_rig"), "b_l_forearm_stub": ("b_l_wrist", "fixed_marker", "forearm_marker")})
  # Optional pose-specific exceptions relax only the axis constraint. The
  # parent/child pivot and rigid bone transforms remain exactly the same.
  if isinstance(relaxed_hinges, str):
    raise TypeError("relaxed_hinges must be a sequence of hinge child bone names.")
  relaxed_hinges = list(relaxed_hinges)
  if len(set(relaxed_hinges)) != len(relaxed_hinges):
    raise ValueError("A relaxed hinge may only be listed once.")
  for name in relaxed_hinges:
    if name not in definitions or definitions[name][1] != "hinge":
      raise ValueError(f"Only an existing hinge may be relaxed: {name}.")
    parent, _, label = definitions[name]
    definitions[name] = (parent, "multi_axis", label)
  children = np.flatnonzero(parents >= 0)
  if {names[i] for i in children} != set(definitions):
    raise ValueError("The joint classification must be reviewed for this rig's bone names.")
  for child in children:
    expected_parent = definitions[names[child]][0]
    if names[parents[child]] != expected_parent:
      raise ValueError(f"Unexpected parent for {names[child]}: expected {expected_parent}.")
  pairs = np.column_stack([parents[children], children]).astype(np.int32)
  kinds = np.asarray([definitions[names[i]][1] for i in children])
  labels = np.asarray([definitions[names[i]][2] for i in children])
  relative = np.linalg.inv(asset["rest_global"][pairs[:, 0]]) @ asset["rest_global"][pairs[:, 1]]
  np.testing.assert_allclose(relative, asset["rest_local"][children], rtol=0.0, atol=1e-12)
  hinge_rows = np.flatnonzero(kinds == "hinge").astype(np.int32)
  hinges = pairs[hinge_rows]
  hinge_rest = relative[hinge_rows]
  # Flexion is local +Z in THIS asset. Parent bind axes differ from child axes.
  axis_child = np.tile([0.0, 0.0, 1.0], (len(hinges), 1))
  axis_parent = np.einsum("hij,hj->hi", hinge_rest[:, :3, :3], axis_child)
  pivot_parent = hinge_rest[:, :3, 3].copy()
  pivot_child = np.zeros_like(pivot_parent)
  child_rest_global = asset["rest_global"][hinges[:, 1]]
  metadata = {"schema_version": np.asarray(1, dtype=np.int32), "bone_names": np.asarray(names), "parents": parents.copy(), "pair_order": np.asarray("parent_child"), "joint_pairs": pairs, "joint_names": labels, "joint_types": kinds, "joint_rest_relative_matrices": relative, "hinge_joint_indices": hinge_rows, "hinge_pairs": hinges, "hinge_pair_names": np.asarray(names)[hinges], "hinge_joint_names": labels[hinge_rows], "hinge_axes_child_local": axis_child, "hinge_axes_parent_local": axis_parent, "hinge_pivots_child_local": pivot_child, "hinge_pivots_parent_local": pivot_parent, "hinge_axes_rest_global": np.einsum("hij,hj->hi", child_rest_global[:, :3, :3], axis_child), "hinge_pivots_rest_global": child_rest_global[:, :3, 3].copy(), "rest_global_matrices": asset["rest_global"].copy(), "length_unit": np.asarray("meter"), "angle_unit": np.asarray("radian"), "hinge_axis_convention": np.asarray("child_bind_local_post_rotation"), "has_angle_limits": np.asarray(False)}
  if relaxed_hinges:
    metadata["schema_version"] = np.asarray(2, dtype=np.int32)
    metadata["relaxed_hinge_names"] = np.asarray(sorted(relaxed_hinges))
    metadata["relaxed_hinge_children"] = np.asarray([names.index(name) for name in sorted(relaxed_hinges)], dtype=np.int32)
  return metadata


def constrained_joint_rotations(target_rotation_vectors, progress, constraints):
  # Only the signed hinge angle is animated. Reject forbidden target rotation
  # rather than silently dropping a user-specified twist or lateral bend.
  from scipy.spatial.transform import Rotation
  children = constraints["hinge_pairs"][:, 1]
  axes = constraints["hinge_axes_child_local"]
  angles = np.einsum("hi,hi->h", target_rotation_vectors[children], axes)
  allowed_vectors = angles[:, None] * axes
  if len(children) and np.max(np.abs(target_rotation_vectors[children] - allowed_vectors)) > 1e-10:
    raise ValueError("A hinge target contains off-axis rotation; only its saved local hinge axis is allowed.")
  fixed = constraints["joint_pairs"][constraints["joint_types"] == "fixed_marker", 1]
  if np.max(np.abs(target_rotation_vectors[fixed])) > 1e-10:
    raise ValueError("A fixed rig marker cannot rotate relative to its parent.")
  vectors = target_rotation_vectors.copy()
  vectors[children] = allowed_vectors
  return Rotation.from_rotvec(float(progress) * vectors).as_matrix()


def hinge_residuals(global_matrices, constraints):
  # G maps each bone's own local points to global coordinates. These affine
  # residuals require no matrix inverse; rigidity is a SEPARATE constraint.
  pairs = constraints["hinge_pairs"]
  parent = global_matrices[..., pairs[:, 0], :, :]
  child = global_matrices[..., pairs[:, 1], :, :]
  parent_pivots = np.einsum("...hij,hj->...hi", parent[..., :3, :3], constraints["hinge_pivots_parent_local"]) + parent[..., :3, 3]
  child_pivots = np.einsum("...hij,hj->...hi", child[..., :3, :3], constraints["hinge_pivots_child_local"]) + child[..., :3, 3]
  parent_axes = np.einsum("...hij,hj->...hi", parent[..., :3, :3], constraints["hinge_axes_parent_local"])
  child_axes = np.einsum("...hij,hj->...hi", child[..., :3, :3], constraints["hinge_axes_child_local"])
  return parent_pivots - child_pivots, parent_axes - child_axes


def rotation_matrices(angles):
  # Local Z flexion, with optional local X/Y components for thumb opposition.
  from scipy.spatial.transform import Rotation
  return Rotation.from_euler("xyz", angles).as_matrix()


def forward_kinematics(rest_local, parents, rotations):
  current = np.empty_like(rest_local)
  for i, parent in enumerate(parents):
    delta = np.eye(4)
    delta[:3, :3] = rotations[i]
    local = rest_local[i] @ delta
    current[i] = current[parent] @ local if parent >= 0 else local
  return current


def skin_vertices(rest_vertices, weights, current, inverse_rest):
  transforms = current @ inverse_rest
  homogeneous = np.column_stack([rest_vertices, np.ones(len(rest_vertices))])
  return np.einsum("vb,bij,vj->vi", weights, transforms[:, :3], homogeneous, optimize=True)


def build_yasps_skinning(model, asset):
  # Bone hierarchy is evaluated through parent JOINs; no copies of animated skin.
  from yasps import attribute
  mesh = model.addMesh("hand")
  parents = asset["parents"]
  levels = np.zeros(len(parents), dtype=np.int32)
  for i, parent in enumerate(parents):
    if parent >= 0:
      levels[i] = levels[parent] + 1
  primitives, updates, order = [], [], []
  previous_lookup = {}
  for depth in range(int(levels.max()) + 1):
    indices = np.flatnonzero(levels == depth)
    bones = mesh.addPrimitive(f"bones_level_{depth}", numInstances=len(indices))
    rest = bones.addConstant("rest_local", rows=4, cols=4)
    rest.updateValue(asset["rest_local"][indices].ravel())
    delta = bones.addAttribute("rotation_delta", rows=4, cols=4)
    delta.updateValue(np.tile(np.eye(4).ravel(), len(indices)))
    inverse_rest = bones.addConstant("inverse_rest_global", rows=4, cols=4)
    inverse_rest.updateValue(np.linalg.inv(asset["rest_global"][indices]).ravel())
    local = rest * delta
    if depth:
      parent_indices = np.asarray([previous_lookup[int(parents[i])] for i in indices], dtype=np.uint32)
      to_parent = bones.addConnectivity("to_parent", primitives[-1], parent_indices.reshape(-1, 1), 1)
      parent_global = bones.addAttribute("parent_global", through=to_parent, source=primitives[-1]["global_matrix"]).resize(4, 4)
      global_matrix = bones.addAttribute("global_matrix", computed_attribute=parent_global * local)
    else:
      global_matrix = bones.addAttribute("global_matrix", computed_attribute=local)
    bones.addAttribute("skinning_matrix", computed_attribute=global_matrix * inverse_rest)
    primitives.append(bones)
    updates.append((indices, delta))
    order.extend(indices.tolist())
    previous_lookup = {int(index): local_index for local_index, index in enumerate(indices)}
  union = mesh.addPrimitiveUnion("all_bones", primitives)
  union.addAttribute("skinning_matrix")
  union.addAttribute("global_matrix")
  original_to_union = np.argsort(order)
  weights = asset["weights"]
  arity = int(np.max(np.count_nonzero(weights, axis=1)))
  influence = np.zeros((len(weights), arity), dtype=np.uint32)
  values = np.zeros((len(weights), arity))
  for i, row in enumerate(weights):
    active = np.flatnonzero(row)
    influence[i, :len(active)] = original_to_union[active]
    values[i, :len(active)] = row[active]
  vertices = mesh.addPrimitive("surface_vertices", numInstances=len(weights))
  rest = vertices.addConstant("rest_homogeneous", rows=4, cols=1)
  rest.updateValue(np.column_stack([asset["vertices"], np.ones(len(weights))]).ravel())
  w = vertices.addConstant("weights", rows=arity, cols=1)
  w.updateValue(values.ravel())
  vertex_to_bones = vertices.addConnectivity("to_bones", union, influence, arity)
  matrices = vertices.addAttribute("bone_matrices", through=vertex_to_bones, source=union["skinning_matrix"])
  deformed = 0.0 * rest
  for j in range(arity):
    deformed = deformed + w[j, 0] * matrices.row(j).resize(4, 4) * rest
  position = vertices.addAttribute("position", computed_attribute=attribute.to_array([deformed[i, 0] for i in range(3)], rows=3, cols=1))
  return updates, position, union["global_matrix"], np.asarray(order)


def update_rotations(updates, rotations):
  for indices, delta in updates:
    matrices = np.broadcast_to(np.eye(4), (len(indices), 4, 4)).copy()
    matrices[:, :3, :3] = rotations[indices]
    delta.updateValue(matrices.ravel())


def build_affine_replay(model, data):
  # This path takes the saved GLOBAL affine transforms directly: no hierarchy
  # evaluation, curl angles, or previously exported surface frames are inputs.
  from yasps import attribute
  mesh = model.addMesh("hand")
  bones = mesh.addPrimitive("affine_bones", numInstances=len(data["bone_names"]))
  matrix = bones.addAttribute("affine_matrix", rows=3, cols=3)
  translation = bones.addAttribute("affine_translation", rows=3, cols=1)
  homogeneous = attribute.to_array([matrix[i, j] if j < 3 else translation[i, 0] for i in range(3) for j in range(4)] + [0, 0, 0, 1], rows=4, cols=4)
  inverse_bind = bones.addConstant("inverse_bind", rows=4, cols=4)
  inverse_bind.updateValue(np.linalg.inv(data["rest_global_matrices"]).ravel())
  skin = bones.addAttribute("skinning_matrix", computed_attribute=homogeneous * inverse_bind)
  weights = data["weights"]
  arity = int(np.max(np.count_nonzero(weights, axis=1)))
  influence = np.zeros((len(weights), arity), dtype=np.uint32)
  values = np.zeros((len(weights), arity))
  for i, row in enumerate(weights):
    active = np.flatnonzero(row)
    influence[i, :len(active)] = active
    values[i, :len(active)] = row[active]
  vertices = mesh.addPrimitive("surface_vertices", numInstances=len(weights))
  rest = vertices.addConstant("rest_position_homogeneous", rows=4, cols=1)
  rest.updateValue(np.column_stack([data["rest_vertices"], np.ones(len(weights))]).ravel())
  w = vertices.addConstant("bone_weights", rows=arity, cols=1)
  w.updateValue(values.ravel())
  connection = vertices.addConnectivity("to_bones", bones, influence, arity)
  skin_matrices = vertices.addAttribute("skin_matrices", through=connection, source=skin)
  result = 0.0 * rest
  for i in range(arity):
    result = result + w[i, 0] * skin_matrices.row(i).resize(4, 4) * rest
  position = vertices.addAttribute("position", computed_attribute=attribute.to_array([result[i, 0] for i in range(3)], rows=3, cols=1))
  return matrix, translation, position


def make_collision_detector(vertices, triangles, ccd_module):
  import pycuda.gpuarray as gpuarray
  edges = np.unique(np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1), axis=0).astype(np.uint32)
  detector = ccd_module.CCD(len(vertices), len(vertices), max_cd_pairs=200_000, max_ccd_pairs=200_000, mesh_indices=np.zeros(len(vertices), dtype=np.uint32), print_timings=False)
  current = gpuarray.to_gpu(vertices.astype(np.float64).ravel())
  direction = gpuarray.zeros_like(current)
  detector.init_faces(current, gpuarray.to_gpu(triangles.astype(np.uint32).ravel()), gpuarray.to_gpu(np.arange(len(vertices), dtype=np.uint32)), len(triangles))
  detector.init_edges(current, current, gpuarray.to_gpu(edges.ravel()), len(edges))
  return detector, current, direction


def safe_segment(detector, gpu_position, gpu_direction, start, end, clearance):
  # CCD convention is x(alpha) = start - alpha * direction.
  gpu_position.set(start.ravel())
  gpu_direction.set((start - end).ravel())
  detector.ccd(gpu_position, clearance * clearance, gpu_direction, 1.0)
  step = detector.compute_largest_step_size(0.9, gpu_position, gpu_direction)
  gpu_position.set(end.ravel())
  near_pairs = sum(detector.cd(gpu_position, clearance * clearance))
  return step >= 1.0 - 1e-10 and near_pairs == 0, float(step), int(near_pairs)


def self_intersections(vertices, triangles):
  # Independent CPU triangle-triangle SAT, after a center/radius broad phase.
  # Incident triangles are excluded: their shared vertices/edges are intentional.
  xyz = vertices[triangles]
  centers = xyz.mean(axis=1)
  radii = np.linalg.norm(xyz - centers[:, None], axis=2).max(axis=1)
  tree = cKDTree(centers)
  pairs = tree.query_pairs(2 * radii.max(), output_type="ndarray")
  if not len(pairs):
    return pairs
  a, b = pairs.T
  keep = np.linalg.norm(centers[a] - centers[b], axis=1) <= radii[a] + radii[b] + 1e-12
  keep &= ~np.any(triangles[a, :, None] == triangles[b, None, :], axis=(1, 2))
  pairs = pairs[keep]
  p, q = xyz[pairs[:, 0]], xyz[pairs[:, 1]]
  keep = np.all(p.max(1) >= q.min(1) - 1e-12, axis=1) & np.all(q.max(1) >= p.min(1) - 1e-12, axis=1)
  pairs, p, q = pairs[keep], p[keep], q[keep]
  e, f = np.roll(p, -1, axis=1) - p, np.roll(q, -1, axis=1) - q
  n, m = np.cross(e[:, 0], e[:, 1]), np.cross(f[:, 0], f[:, 1])
  axes = np.concatenate([n[:, None], m[:, None], np.cross(e[:, :, None], f[:, None]).reshape(-1, 9, 3), np.cross(n[:, None], e), np.cross(m[:, None], f)], axis=1)
  norm = np.linalg.norm(axes, axis=2)
  axes = axes / np.maximum(norm[:, :, None], 1e-30)
  pp, qq = np.einsum("nvc,nac->nva", p, axes), np.einsum("nvc,nac->nva", q, axes)
  separated = (pp.max(1) < qq.min(1) - 1e-11) | (qq.max(1) < pp.min(1) - 1e-11)
  separated &= norm > 1e-20
  return pairs[~np.any(separated, axis=1)]


def save_video(output_dir, num_frames, fps):
  import subprocess
  path = output_dir / "hand_open_to_fist.mp4"
  subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(output_dir / "frame_%04d.jpg"), "-frames:v", str(num_frames), "-c:v", "libx264", "-threads", "8", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], check=True)
  return path
