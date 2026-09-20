"""Geometry, transfer references, symbolic energies, and safeguards for this example.

NumPy is used for geometry/transfers and independent tests, never to differentiate
or assemble a simulation Hessian. The optimization graphs are all YASPS graphs.
"""
from pathlib import Path
import subprocess

import numpy as np


GRID_N = 64
BOX_WIDTH = 0.5
BOX_LOWER = np.array([-BOX_WIDTH / 2, 0.0, -BOX_WIDTH / 2])
BOX_UPPER = BOX_LOWER + BOX_WIDTH
# N nodes span (N-1) intervals: the box plus half an interval at each end.
DX = BOX_WIDTH / (GRID_N - 2)
GRID_ORIGIN = BOX_LOWER - DX / 2
GRID = GRID_ORIGIN + DX * np.indices((GRID_N,) * 3).reshape(3, -1).T
OFFSETS = np.indices((3, 3, 3)).reshape(3, -1).T
TRANSFER_OPTIONS = ["-std=c++17", f"-DGRID_N={GRID_N}", f"-DGRID_DX={DX!r}"] + [f"-DGRID_ORIGIN_{axis}={float(value)!r}" for axis, value in zip("XYZ", GRID_ORIGIN)]


def create_collision_detector(geometry, faces, edges, query_ids, mesh_ids, capacity, cd_capacity=20_000_000):
  from ccd import CCD
  import pycuda.gpuarray as gpuarray
  detector = CCD(len(query_ids), len(mesh_ids), max_cd_pairs=cd_capacity, max_ccd_pairs=capacity, mesh_indices=mesh_ids, print_timings=False)
  detector.init_faces(geometry, gpuarray.to_gpu(faces.ravel()), gpuarray.to_gpu(query_ids), len(faces))
  detector.init_edges(geometry, geometry, gpuarray.to_gpu(edges.ravel()), len(edges))
  return detector


def ccd_sweep_with_growth(detector, capacity, geometry, distance_squared, direction, alpha, faces, edges, query_ids, mesh_ids, cd_capacity=20_000_000):
  # Retry the identical sweep after capacity growth; do not change the step.
  while True:
    try:
      detector.ccd(geometry, distance_squared, direction, alpha)
      return detector, capacity
    except OverflowError as error:
      if not str(error).startswith("CCD broad-phase candidate capacity exceeded:"):
        raise
      previous = capacity
      capacity = (3 * capacity + 1) // 2
      detector.close()  # Releases old GPU buffers before allocating larger ones.
      print(f"CCD CAPACITY GROW {previous} -> {capacity}; retrying alpha={alpha:.6e}", flush=True)
      detector = create_collision_detector(geometry, faces, edges, query_ids, mesh_ids, capacity, cd_capacity)


##################################################################
## Positive-volume geometry and reproducible volume sampling
##################################################################
def tet_volumes(vertices, tets):
  p = vertices[tets]
  return np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]) / 6.0


def load_bunny(data_directory):
  with open(Path(data_directory) / "bunny.node") as stream:
    next(stream)
    vertices = np.array([[float(x) for x in line.split()[1:4]] for line in stream if line.strip() and not line.startswith("#")])
  with open(Path(data_directory) / "bunny.ele") as stream:
    next(stream)
    tets = np.array([[int(x) - 1 for x in line.split()[3:7]] for line in stream if line.strip() and not line.startswith("#")], dtype=np.uint32)
  negative = tet_volumes(vertices, tets) < 0.0
  tets[negative, 1], tets[negative, 2] = tets[negative, 2].copy(), tets[negative, 1].copy()
  if np.any(tet_volumes(vertices, tets) <= 0):
    raise ValueError("Degenerate bunny tetrahedron")
  return vertices, tets


def surface_triangles(tets):
  faces = tets[:, [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]].reshape(-1, 3)
  _, first, counts = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
  return faces[first[counts == 1]].astype(np.uint32)


def surface_edges(triangles):
  edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
  return np.unique(np.sort(edges, axis=1), axis=0).astype(np.uint32)


def vertex_masses(vertices, tets, density):
  volumes = tet_volumes(vertices, tets)
  masses = np.zeros(len(vertices))
  np.add.at(masses, tets.ravel(), np.repeat(density * volumes / 4.0, 4))
  return masses, volumes


def volume_samples(vertices, tets, count, rng):
  volumes = tet_volumes(vertices, tets)
  chosen = rng.choice(len(tets), count, p=volumes / volumes.sum())
  # Dirichlet(1,1,1,1), not normalized uniform random barycentric weights.
  barycentric = rng.exponential(size=(count, 4))
  barycentric /= barycentric.sum(axis=1, keepdims=True)
  return np.einsum("ni,nij->nj", barycentric, vertices[tets[chosen]]), np.full(count, volumes.sum() / count)


def rotation_xyz(degrees):
  x, y, z = np.deg2rad(degrees)
  rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
  ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
  rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
  return rz @ ry @ rx


def place_bunny(vertices, tets, size, center, angles):
  masses, _ = vertex_masses(vertices, tets, 1.0)
  rest = (vertices - np.average(vertices, axis=0, weights=masses)) * (size / np.ptp(vertices, axis=0).max())
  rotation = rotation_xyz(angles)
  rotated = rest @ rotation.T
  translation = np.array(center) - 0.5 * (rotated.min(axis=0) + rotated.max(axis=0))
  return rest, rotated + translation, np.column_stack([rotation, translation])


def check_aabbs(bodies, margin=0.003):
  for i, a in enumerate(bodies):
    for j, b in enumerate(bodies[:i]):
      if np.all(a.min(axis=0) < b.max(axis=0) + margin) and np.all(b.min(axis=0) < a.max(axis=0) + margin):
        raise ValueError(f"Initial transformed bunny AABBs overlap: {j}, {i}")


def container_mesh():
  corners = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]])
  vertices = BOX_LOWER + corners * (BOX_UPPER - BOX_LOWER)
  quads = np.array([[0, 3, 2, 1], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])
  return vertices, quads[:, [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3).astype(np.uint32)


##################################################################
## Independent quadratic APIC reference (also useful for debugging)
##################################################################
def stencil(x):
  xi = (x - GRID_ORIGIN) / DX
  base = np.floor(xi - 0.5).astype(np.int64)
  if np.any(base < 0) or np.any(base + 2 >= GRID_N):
    raise ValueError("Particle support leaves the grid; no indices/weights were clamped")
  f = xi - base
  w = np.stack([0.5 * (1.5 - f)**2, 0.75 - (f - 1)**2, 0.5 * (f - 0.5)**2], axis=1)
  dw = np.stack([f - 1.5, -2 * (f - 1), f - 0.5], axis=1) / DX
  selected = w[:, OFFSETS, np.arange(3)]
  selected_dw = dw[:, OFFSETS, np.arange(3)]
  weights = selected.prod(axis=2)
  gradients = np.stack([selected_dw[:, :, a] * selected[:, :, (a + 1) % 3] * selected[:, :, (a + 2) % 3] for a in range(3)], axis=2)
  nodes = base[:, None, :] + OFFSETS[None]
  indices = ((nodes[:, :, 0] * GRID_N + nodes[:, :, 1]) * GRID_N + nodes[:, :, 2]).astype(np.uint32)
  return indices, weights, gradients


def p2g_reference(x, v, F, C, mass):
  indices, weights, gradients = stencil(x)
  offsets = GRID[indices] - x[:, None]
  local_velocity = v[:, None] + np.einsum("pab,pib->pia", C, offsets)
  wm = mass[:, None] * weights
  grid_mass = np.bincount(indices.ravel(), weights=wm.ravel(), minlength=len(GRID))
  momentum = np.stack([np.bincount(indices.ravel(), weights=(wm * local_velocity[:, :, a]).ravel(), minlength=len(GRID)) for a in range(3)], axis=1)
  velocity = np.zeros_like(momentum)
  np.divide(momentum, grid_mass[:, None], out=velocity, where=grid_mass[:, None] > 0)
  B = np.einsum("pia,pab->pib", gradients, F)
  return indices, weights, gradients, B, grid_mass, velocity


def trial_reference(q, x, F, indices, weights, B):
  displacement = q[indices] - GRID[indices]
  return x + np.einsum("pi,pia->pa", weights, displacement), F + np.einsum("pia,pib->pab", displacement, B)


def g2p_reference(q, h, x, F, indices, weights, B):
  position, deformation = trial_reference(q, x, F, indices, weights, B)
  grid_velocity = (q[indices] - GRID[indices]) / h
  velocity = np.einsum("pi,pia->pa", weights, grid_velocity)
  C = 4.0 / DX**2 * np.einsum("pi,pia,pib->pab", weights, grid_velocity, GRID[indices] - x[:, None])
  return position, velocity, deformation, C


##################################################################
## Symbolic energies: physical densities have no implicit h or volume
##################################################################
def snh_parameters(young, poisson):
  mu = young / (2.0 * (1.0 + poisson))
  lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
  return 4.0 * mu / 3.0, lam + 5.0 * mu / 6.0


def snh_density(F, mu, lam):
  invariant = (F.transpose() * F).trace()
  jacobian = F.determinant()
  a = 1.0 + 0.75 * mu / lam
  return 0.5 * mu * (invariant - 3.0) - 0.5 * mu * ((invariant + 1.0) / 4.0).log() + 0.5 * lam * ((jacobian - a) * (jacobian - a) - (1.0 - a) * (1.0 - a))


def determinant_barrier(jacobian, stiffness, activation, minimum, stiffness_scale):
  from yasps import attribute
  normalized = (jacobian - minimum) / (activation - minimum)
  safe_normalized = attribute.select(normalized > attribute(float_value=1e-12), normalized, attribute(float_value=1e-12))
  residual = jacobian - activation
  active = -stiffness_scale * stiffness * residual * residual * safe_normalized.log()
  return attribute.select(jacobian < activation, active, attribute(float_value=0.0))


def liquid_density(F, bulk):
  difference = F.determinant() - 1.0
  return 0.5 * bulk * difference * difference


def cofactor_matrices(F):
  """Return d(det(F))/dF for a batch of row-major 3x3 matrices."""
  F = np.asarray(F, dtype=np.float64).reshape(-1, 3, 3)
  return np.stack([np.cross(F[:, 1], F[:, 2]), np.cross(F[:, 2], F[:, 0]), np.cross(F[:, 0], F[:, 1])], axis=1)


def snh_spectral_data(F, mu, lam):
  """Freeze the exact value, gradient, and spectral Hessian blocks of SNH."""
  F = np.asarray(F, dtype=np.float64).reshape(-1, 3, 3)
  mu = np.broadcast_to(np.asarray(mu, dtype=np.float64).reshape(-1), len(F))
  lam = np.broadcast_to(np.asarray(lam, dtype=np.float64).reshape(-1), len(F))
  U, sigma, Vh = np.linalg.svd(F)
  V = Vh.transpose(0, 2, 1)
  flip = np.linalg.det(U) < 0.0
  U[flip, :, 2] *= -1.0
  sigma[flip, 2] *= -1.0
  flip = np.linalg.det(V) < 0.0
  V[flip, :, 2] *= -1.0
  sigma[flip, 2] *= -1.0

  squared_norm = np.einsum("ni,ni->n", sigma, sigma)
  q = squared_norm + 1.0
  jacobian = np.prod(sigma, axis=1)
  a = 1.0 + 0.75 * mu / lam
  cof_sigma = np.column_stack([sigma[:, 1] * sigma[:, 2], sigma[:, 0] * sigma[:, 2], sigma[:, 0] * sigma[:, 1]])
  psi = 0.5 * mu * (squared_norm - 3.0) - 0.5 * mu * np.log(q / 4.0) + 0.5 * lam * ((jacobian - a)**2 - (1.0 - a)**2)
  gradient = mu[:, None] * (1.0 - 1.0 / q[:, None]) * sigma + lam[:, None] * (jacobian - a)[:, None] * cof_sigma

  B = 2.0 * mu[:, None, None] * sigma[:, :, None] * sigma[:, None, :] / q[:, None, None]**2
  B[:, np.arange(3), np.arange(3)] += mu[:, None] * (1.0 - 1.0 / q[:, None])
  B += lam[:, None, None] * cof_sigma[:, :, None] * cof_sigma[:, None, :]
  for i, j, remaining in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
    correction = lam * (jacobian - a) * sigma[:, remaining]
    B[:, i, j] += correction
    B[:, j, i] += correction

  beta_sym = np.empty((len(F), 3))
  beta_skew = np.empty((len(F), 3))
  for pair, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
    difference = sigma[:, i] - sigma[:, j]
    total = sigma[:, i] + sigma[:, j]
    difference_limit = 1e-8 * np.maximum(1.0, np.maximum(abs(sigma[:, i]), abs(sigma[:, j])))
    total_limit = 1e-8 * np.maximum(1.0, np.maximum(abs(sigma[:, i]), abs(sigma[:, j])))
    beta_sym[:, pair] = np.where(abs(difference) > difference_limit, (gradient[:, i] - gradient[:, j]) / np.where(difference != 0.0, difference, 1.0), 0.5 * (B[:, i, i] + B[:, j, j] - B[:, i, j] - B[:, j, i]))
    beta_skew[:, pair] = np.where(abs(total) > total_limit, (gradient[:, i] + gradient[:, j]) / np.where(total != 0.0, total, 1.0), 0.5 * (B[:, i, i] + B[:, j, j] + B[:, i, j] + B[:, j, i]))
  return {"U": U, "V": V, "psi": psi, "g_sigma": gradient, "B_sigma": B, "beta_sym": np.maximum(beta_sym, 0.0), "beta_skew": np.maximum(beta_skew, 0.0)}


def freeze_particle_model(group):
  """Freeze the liquid volume ratio for the entire Newton solve, on the GPU."""
  if group["liquid"]:
    group["frozen"]["J"] = group["state"]["J"]


def liquid_volume_from_deformation(values):
  """Migrate an old checkpoint once; new checkpoints store liquid J directly.

  Decimal avoids cancellation when the stored F is already badly sheared.
  This recovers det(stored F), not accuracy lost before the checkpoint; signs
  and magnitudes are preserved without clamping or resetting volume.
  """
  from decimal import Decimal, localcontext
  matrices = np.asarray(values, dtype=np.float64).reshape(-1, 9)
  volumes = np.empty(len(matrices), dtype=np.float64)
  with localcontext() as ctx:
    ctx.prec = 80
    for index, matrix in enumerate(matrices):
      a, b, c, d, e, f, g, h, i = [Decimal.from_float(float(x)) for x in matrix]
      volumes[index] = float(a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g))
  return volumes


def inertia(position, target, mass):
  difference = position - target
  return 0.5 * mass * difference.dot(difference)


def edge_matrix(positions):
  from yasps import attribute
  return attribute.to_array([positions[b + 1, a] - positions[0, a] for a in range(3) for b in range(3)], rows=3, cols=3)


def rotation_penalty(T, orth_weight, det_weight):
  from yasps import attribute
  A = attribute.to_array([T[a, b] for a in range(3) for b in range(3)], rows=3, cols=3)
  error = A.transpose() * A - attribute.identity(3)
  orth = sum(error[a, b] * error[a, b] for a in range(3) for b in range(3))
  det_error = A.determinant() - 1.0
  return 0.5 * orth_weight * orth + 0.5 * det_weight * det_error * det_error


def contact_distance_squared(position, kind):
  p = [position.row(i) for i in range(position.rows)]
  if kind == "pp":
    d = p[1] - p[0]
    return d.dot(d)
  if kind == "pe":
    cross = (p[1] - p[0]).cross(p[2] - p[0])
    return cross.dot(cross) / (p[2] - p[1]).dot(p[2] - p[1])
  if kind == "pt":
    normal = (p[2] - p[1]).cross(p[3] - p[1])
    numerator = (p[0] - p[1]).dot(normal)
  else:
    normal = (p[1] - p[0]).cross(p[3] - p[2])
    numerator = (p[2] - p[0]).dot(normal)
  return numerator * numerator / normal.dot(normal)


def contact_barrier(position, kind, dhat, coefficient):
  # Classification/refiltering supplies ONLY s < dhat, exactly as the
  # branch's mixed-separation PP/PE/PT/EE helpers (no extra mollifier there).
  s = contact_distance_squared(position, kind)
  difference = s - dhat
  logarithm = (s / dhat).log()
  return coefficient * difference * difference * logarithm * logarithm


def axis_aligned_wall_barriers(position, dhat, coefficient, lower, upper):
  """Five independent IPC-style barriers for an open-top box.

  Each signed gap is affine in one position coordinate. On the feasible side,
  the active scalar barrier is convex, so its spatial Hessian is a PSD
  rank-one matrix along the wall normal and has no tangential contribution.
  Returning separate expressions also keeps each generated derivative kernel
  small instead of asking CUDA to optimize a large five-branch expression.
  """
  from yasps import attribute
  lower_x, lower_y, lower_z = map(float, lower)
  upper_x, upper_z = float(upper[0]), float(upper[2])
  gaps = [("x_min", position[0] - lower_x), ("x_max", upper_x - position[0]), ("y_min", position[1] - lower_y), ("z_min", position[2] - lower_z), ("z_max", upper_z - position[2])]
  barriers = []
  for name, gap in gaps:
    squared_distance = gap * gap
    difference = squared_distance - dhat
    # Do not clamp the log argument: that removes the repulsive derivative
    # near the wall and makes the Hessian indefinite. The feasible domain is
    # strictly positive gap; zero gap must remain an infinite barrier.
    logarithm = (squared_distance / dhat).log()
    active = coefficient * difference * difference * logarithm * logarithm
    barriers.append((name, attribute.select(squared_distance < dhat, active, attribute(float_value=0.0))))
  return barriers


def add_particle_walls(mesh, group, dhat, coefficient, lower, upper, transfer, distance_squared):
  # Each selected particle position is a JOIN boundary: differentiate the
  # wall locally in 3 coordinates, then propagate through the 27-node stencil.
  import pycuda.gpuarray as gpuarray
  count = group["count"]
  group["walls"] = []
  group["wall_filter"] = transfer.get_function("select_particle_walls")
  group["add_wall_blocks"] = transfer.get_function("add_particle_wall_blocks")
  group["wall_capacity"] = count
  group["wall_ids"] = gpuarray.empty(5 * count, np.uint32)
  group["wall_weights"] = gpuarray.empty(5 * count, np.float64)
  group["wall_counts"] = gpuarray.zeros(5, np.uint32)
  group["wall_bounds"] = tuple(np.float64(x) for x in (lower[0], upper[0], lower[1], lower[2], upper[2]))
  group["wall_distance_squared"] = np.float64(distance_squared)
  group["wall_batch_first"] = 0
  for wall_index, wall_name in enumerate(("x_min", "x_max", "y_min", "z_min", "z_max")):
    wall = mesh.addPrimitive(f"{group['primitive'].name}_wall_{wall_name}", numInstances=0, isDynamic=True)
    connection = wall.addConnectivity("particle", group["primitive"], np.empty(0, np.uint32), 1)
    position = wall.addAttribute("position", through=connection, source=group["position"]).resize(1, 3)
    weight = constant(wall, "weight", np.empty(0))
    expression = axis_aligned_wall_barriers(position, dhat, coefficient * weight, lower, upper)[wall_index][1]
    energy = wall.addAttribute("energy", computed_attribute=expression)
    group["walls"].append({"primitive": wall, "connection": connection, "weight": weight, "energy": energy, "name": wall_name})
    group["energies"].append(energy)
    group["energy_groups"].append(([energy], -1, f"analytic_wall_{wall_name}"))


def update_particle_walls(group):
  # Positions come from YASPS. Only the geometric selection and coefficient
  # gathering happen here; energy/gradient/Hessian evaluation stays in YASPS.
  if not group.get("walls"):
    return
  count = group["primitive"].numInstances
  first = group["wall_batch_first"]
  group["wall_counts"].fill(0)
  if count:
    position = group["position"].compute().value
    group["wall_filter"](position, group["frozen"]["wall_weight"][first:first + count], group["wall_ids"], group["wall_weights"], group["wall_counts"], np.uint32(count), np.uint32(group["wall_capacity"]), group["wall_distance_squared"], *group["wall_bounds"], block=(128, 1, 1), grid=((count + 127) // 128, 1, 1))
  counts = group["wall_counts"].get()
  for index, wall in enumerate(group["walls"]):
    active = int(counts[index])
    start = index * group["wall_capacity"]
    wall["primitive"].updateNumInstances(active)
    # Zero instances need no entries. Retain existing capacity, as for CCD
    # contacts, rather than issuing a copy through a null zero-length buffer.
    if active:
      wall["connection"].updateConnectivity(group["wall_ids"][start:start + active])
      wall["weight"].updateValue(group["wall_weights"][start:start + active])


##################################################################
## Keep particle interpolation stencils inside the fixed grid
##################################################################
def domain_step_bound(x, direction, safety=0.9):
  lower = GRID_ORIGIN + (0.5 + 1e-9) * DX
  upper = GRID_ORIGIN + (GRID_N - 1.5 - 1e-9) * DX
  if np.any(x < lower) or np.any(x > upper):
    return 0.0
  distance = np.where(direction > 0, upper - x, x - lower)
  limits = np.full_like(x, np.inf)
  np.divide(distance, np.abs(direction), out=limits, where=direction != 0)
  crossing = limits.min(initial=np.inf)
  return 1.0 if crossing > 1.0 else max(0.0, safety * crossing)


def save_video(directory, frames, fps=100, first_frame=0):
  directory = Path(directory)
  output = directory / "mpm_fem_affine_bunnies.mp4"
  subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-start_number", str(first_frame), "-i", str(directory / "frame_%04d.jpg"), "-frames:v", str(frames), "-c:v", "libx264", "-threads", "8", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)], check=True)
  return output


##################################################################
## Fixed-arity YASPS particle graphs and memory-bounded native assembly
##################################################################
def constant(primitive, name, values, rows=1, cols=1):
  result = primitive.addConstant(name, rows=rows, cols=cols)
  result.updateValue(np.asarray(values, dtype=np.float64).ravel())
  return result


def position_attributes(primitive, values, separate):
  from yasps import attribute
  if separate:
    coordinates = [primitive.addAttribute(axis) for axis in "xyz"]
    position = primitive.addAttribute("position", computed_attribute=attribute.to_array(coordinates, rows=1, cols=3))
  else:
    position = primitive.addAttribute("position", rows=1, cols=3)
    coordinates = [position]
  update_position(coordinates, values)
  return coordinates, position


def update_position(coordinates, values):
  # Preserve the interleaved physical checkpoint/transfer format. Only the
  # independent YASPS unknowns are axis-major in the scalar formulation.
  values = values.reshape(-1)
  if len(coordinates) == 1:
    coordinates[0].updateValue(values, deepCopy=True)
  else:
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
    for axis, coordinate in enumerate(coordinates):
      # GPUArray.copy preserves strides; explicitly gather contiguous scalar
      # storage because generated attribute kernels do not accept strides.
      if isinstance(values, np.ndarray):
        value = values[axis::3].copy()
      else:
        value = gpuarray.empty(values.size // 3, values.dtype)
        copy = cuda.Memcpy2D()
        copy.set_src_device(int(values.gpudata) + axis * values.dtype.itemsize)
        copy.set_dst_device(value.gpudata)
        copy.src_pitch = 3 * values.dtype.itemsize
        copy.dst_pitch = copy.width_in_bytes = values.dtype.itemsize
        copy.height = value.size
        copy(aligned=False)
      coordinate.updateValue(value, deepCopy=True)


def particle_graph(mesh, name, grid, q, grid_rest, count, h, liquid, determinant_activation, minimum_determinant, determinant_barrier_scale, liquid_determinant_weight=None):
  from yasps import attribute
  particles = mesh.addPrimitive(name, numInstances=count, isDynamic=True)
  connection = particles.addConnectivity("stencil", grid, np.tile(np.arange(27, dtype=np.uint32), (count, 1)), 27)
  old_x = constant(particles, "old_x", np.zeros((count, 3)), 1, 3)
  old_F = constant(particles, "old_F", np.tile(np.eye(3), (count, 1, 1)), 3, 3)
  weights = constant(particles, "weights", np.zeros((count, 27)), 27)
  # Reuse the stencil buffer: liquid stores grad(w), solid stores grad(w)*F0.
  B = constant(particles, "shape_gradients" if liquid else "B", np.zeros((count, 27, 3)), 27, 3)
  if isinstance(q, list):
    displacement = []
    coordinates = []
    for axis, axis_name in enumerate("xyz"):
      values = particles.addAttribute(f"grid_{axis_name}", through=connection, source=q[axis]).resize(27, 1)
      reference = particles.addAttribute(f"grid_reference_{axis_name}", through=connection, source=grid_rest[axis]).resize(27, 1)
      delta = values - reference
      displacement.append(delta)
      coordinates.append(particles.addAttribute(axis_name, computed_attribute=old_x[axis] + (weights.transpose() * delta)[0]))
    position = particles.addAttribute("position", computed_attribute=attribute.to_array(coordinates, rows=1, cols=3))
    increment = attribute.to_array([(displacement[axis].transpose() * B)[j] for axis in range(3) for j in range(3)], rows=3, cols=3)
  else:
    grid_reference = particles.addAttribute("grid_reference", through=connection, source=grid_rest)
    grid_values = particles.addAttribute("grid_position", through=connection, source=q)
    displacement = grid_values - grid_reference
    position = particles.addAttribute("position", computed_attribute=old_x + weights.transpose() * displacement)
    increment = displacement.transpose() * B
  # Liquid F is retained only for export/diagnostics, not its volume energy.
  F = particles.addAttribute("F", computed_attribute=old_F + (increment * old_F if liquid else increment))
  bindings = [(old_x, "x", 3), (old_F, "F", 9), (weights, "weights", 27), (B, "B", 81)]
  materials, energies = [], []

  if liquid and liquid_determinant_weight is None:
    material = mesh.addPrimitive(f"{name}_model", numInstances=count, isDynamic=True)
    identity = material.addConnectivity("particle", particles, np.arange(count, dtype=np.uint32), 1)
    J = constant(particles, "volume_ratio", np.ones(count))
    # cof(F0):(D*F0) = det(F0)*trace(D), without large cancelling products.
    divergence = sum(increment[axis, axis] for axis in range(3))
    residual = particles.addAttribute("volume_residual", computed_attribute=J - 1.0 + J * divergence)
    # The one-to-one join preserves a scalar inner Hessian, without projection.
    local_residual = material.addAttribute("residual", through=identity, source=residual)
    volume = constant(material, "volume", np.ones(count))
    bulk = constant(material, "bulk", np.ones(count))
    density = 0.5 * bulk * local_residual * local_residual
    energy = material.addAttribute("energy", computed_attribute=h * h * volume * density)
    bindings += [(J, "J", 1), (volume, "volume", 1), (bulk, "bulk", 1)]
    materials.append(material)
    energies.append(energy)
  elif not liquid:
    material = mesh.addPrimitive(f"{name}_model", numInstances=count, isDynamic=True)
    identity = material.addConnectivity("particle", particles, np.arange(count, dtype=np.uint32), 1)
    local_F = material.addAttribute("F", through=identity, source=F).resize(3, 3)
    volume = constant(material, "volume", np.ones(count))
    mu = constant(material, "mu", np.ones(count))
    lam = constant(material, "lam", np.ones(count))
    density = snh_density(local_F, mu, lam)
    energy = material.addAttribute("energy", computed_attribute=h * h * volume * density)
    bindings += [(volume, "volume", 1), (mu, "mu", 1), (lam, "lam", 1)]
    materials.append(material)
    energies.append(energy)

  energy_groups = ([(energies.copy(), -1, "liquid_gn")] if liquid else [(energies.copy(), 1, "solid_stable_neo_hookean")]) if energies else []
  if liquid and liquid_determinant_weight is not None:
    # Join F so the exact determinant penalty has a 9x9 inner Hessian.
    # This replaces the frozen GN energy when an exact determinant weight is supplied.
    material = mesh.addPrimitive(f"{name}_determinant_model", numInstances=count, isDynamic=True)
    identity = material.addConnectivity("particle", particles, np.arange(count, dtype=np.uint32), 1)
    local_F = material.addAttribute("F", through=identity, source=F).resize(3, 3)
    # Expand det(F) so autodiff stays polynomial even when F is singular.
    determinant = local_F[0, 0] * (local_F[1, 1] * local_F[2, 2] - local_F[1, 2] * local_F[2, 1]) - local_F[0, 1] * (local_F[1, 0] * local_F[2, 2] - local_F[1, 2] * local_F[2, 0]) + local_F[0, 2] * (local_F[1, 0] * local_F[2, 1] - local_F[1, 1] * local_F[2, 0])
    determinant = material.addAttribute("determinant", computed_attribute=determinant)
    difference = determinant - 1.0
    energy = material.addAttribute("energy", computed_attribute=liquid_determinant_weight * difference * difference)
    materials.append(material)
    energies.append(energy)
    energy_groups.append(([energy], 2, "liquid_exact_determinant_penalty"))
  return {"primitive": particles, "material": materials[0], "materials": materials, "connection": connection, "count": count, "position": position, "F": F, "energy": energies[0], "energies": energies, "energy_groups": energy_groups, "liquid": liquid, "bindings": bindings}


def bind_particle_range(group, first=0, last=None):
  # Bind the current particle DATA and frozen grid stencil to the YASPS graph.
  last = group["count"] if last is None else last
  group["primitive"].updateNumInstances(last - first)
  for material in group["materials"]:
    material.updateNumInstances(last - first)
  group["connection"].updateConnectivity(group["frozen"]["indices"][first * 27:last * 27])
  for att, key, width in group["bindings"]:
    att.updateValue(group["frozen"][key][first * width:last * width])
  group["wall_batch_first"] = first
  update_particle_walls(group)


def copy_block_topology(H, transfer, suffix=""):
  """Copy occupied block coordinates and expand shape metadata on the GPU."""
  import pycuda.gpuarray as gpuarray
  counts = getattr(H, f"block_counts{suffix}")
  shapes = getattr(H, f"block_dimensions{suffix}")
  count = sum(counts)
  coordinates = getattr(H, f"block_positions{suffix}")[:2 * count].copy() if count else gpuarray.empty(0, np.uint32)
  dimensions = gpuarray.empty(2 * count, np.uint32)
  cursor = 0
  for shape_index, shape_count in enumerate(counts):
    if shape_count:
      rows, cols = shapes[2 * shape_index:2 * shape_index + 2]
      transfer.get_function("topology_block_dimensions")(
        dimensions, np.uint32(cursor), np.uint32(shape_count), np.uint32(rows), np.uint32(cols),
        block=(128, 1, 1), grid=((shape_count + 127) // 128, 1, 1))
    cursor += shape_count
  return coordinates, dimensions


class FrameHierarchyTopology:
  """Static, frozen-particle and optional initial-contact topology.

  P2G emits each 27-node support in the same offset order, so its first node
  uniquely identifies the *whole* stencil. A GPU representative table therefore
  removes repeated supports exactly before YASPS generates any pair coordinates.
  Coordinates still come from the real differentiated Hessian's index kernels;
  material parameters, weights, and Hessian values are never evaluated here.
  """

  def __init__(self, static_hessian, groups, batch_size, transfer):
    import pycuda.gpuarray as gpuarray
    from pycuda.scan import InclusiveScanKernel
    if batch_size <= 0:
      raise ValueError("Topology batch size must be positive")
    self.groups = groups
    self.batch_size = batch_size
    self.transfer = transfer
    # Deliberately omit the scene's dynamic collision coordinates. Static
    # inertia diagonals retain every grid, soft, and affine variable block.
    self.static_coordinates, self.static_dimensions = copy_block_topology(static_hessian, transfer)
    self.representatives = gpuarray.empty(len(GRID), np.uint32)
    self.representative_ordinals = gpuarray.empty(len(GRID), np.uint32)
    self.scan = InclusiveScanKernel(np.uint32, "a+b")
    self.block_positions = self.block_dimensions = None
    self.num_blocks = 0
    self.contact_block_count = 0
    self.unique_stencil_counts = []

  def _unique_stencils(self, group):
    import pycuda.gpuarray as gpuarray
    self.representatives.fill(np.uint32(0xffffffff))
    if group["count"]:
      self.transfer.get_function("topology_stencil_representatives")(
        group["frozen"]["indices"], self.representatives, np.uint32(group["count"]),
        block=(128, 1, 1), grid=((group["count"] + 127) // 128, 1, 1))
    self.transfer.get_function("topology_stencil_marks")(
      self.representatives, self.representative_ordinals,
      block=(128, 1, 1), grid=((len(GRID) + 127) // 128, 1, 1))
    self.scan(self.representative_ordinals)
    count = int(self.representative_ordinals[-1:].get()[0])
    indices = gpuarray.empty(count * 27, np.uint32)
    if count:
      self.transfer.get_function("topology_gather_stencils")(
        group["frozen"]["indices"], self.representatives, self.representative_ordinals, indices,
        block=(128, 1, 1), grid=((len(GRID) + 127) // 128, 1, 1))
    return indices, count

  def collect(self, contact_hessian=None):
    """Collect GPU arrays with bounded raw coordinate work per batch."""
    import pycuda.gpuarray as gpuarray
    coordinates, dimensions = [self.static_coordinates], [self.static_dimensions]
    self.unique_stencil_counts = []
    for group in self.groups:
      representative_indices, count = self._unique_stencils(group)
      self.unique_stencil_counts.append(count)
      try:
        # Material stencils already contain every possible wall coupling.
        # Do not use full-particle wall IDs with representative-only bindings.
        for wall in group.get("walls", []):
          wall["primitive"].updateNumInstances(0)
        for first in range(0, count, self.batch_size):
          last = min(first + self.batch_size, count)
          # Index kernels only need instance counts and connectivity, not
          # gathered particle coefficients. Restore all bindings below before
          # any energy, geometry, or numerical Hessian evaluation can run.
          group["primitive"].updateNumInstances(last - first)
          for material in group["materials"]:
            material.updateNumInstances(last - first)
          group["connection"].updateConnectivity(representative_indices[27 * first:27 * last])
          for H in group["hessians"]:
            if group.get("material_hessian") is not None and H is not group["material_hessian"]:
              continue
            H.getSparseIndicesDynamicAgain()
            position, dimension = copy_block_topology(H, self.transfer, "_dynamic")
            coordinates.append(position)
            dimensions.append(dimension)
      finally:
        bind_particle_range(group)
    # All particle bindings must be restored before the mixed contact union's
    # indices are evaluated. Only generate coordinates, never Hessian values.
    self.contact_block_count = 0
    if contact_hessian is not None:
      contact_hessian.getSparseIndicesDynamicAgain()
      position, dimension = copy_block_topology(contact_hessian, self.transfer, "_dynamic")
      self.contact_block_count = position.size // 2
      if self.contact_block_count:
        coordinates.append(position)
        dimensions.append(dimension)
    # These inputs contain compressed coordinates, not particle-count times
    # all 27x27 raw local pairs. Repeated edges between batches are valid input
    # to the hierarchy API, which canonicalizes the complete frame graph.
    self.block_positions = gpuarray.concatenate(coordinates)
    self.block_dimensions = gpuarray.concatenate(dimensions)
    self.num_blocks = self.block_positions.size // 2
    return self.block_positions, self.block_dimensions, self.num_blocks

  def apply(self, solver):
    """Install the collected frame graph, also after an exceptional reset."""
    if self.block_positions is None:
      raise RuntimeError("Collect frame topology before rebuilding its hierarchy")
    solver.rebuildHierarchy(self.block_positions, self.block_dimensions, self.num_blocks)

  def rebuild(self, solver, contact_hessian=None):
    self.collect(contact_hessian)
    self.apply(solver)


def copy_block_part(H, suffix="_dynamic"):
  # Copy only occupied compressed sparse blocks; no dense local Hessians are
  # retained between batches.
  pieces = []
  cursor = 0
  dims = getattr(H, f"block_dimensions{suffix}")
  for c, count in enumerate(getattr(H, f"block_counts{suffix}")):
    if count:
      rows, cols = dims[2 * c:2 * c + 2]
      start = getattr(H, f"blocks_start_indices{suffix}")[c]
      values = getattr(H, f"blocks_flattened{suffix}")[start:start + count * rows * cols].copy()
      coordinates = getattr(H, f"block_positions{suffix}")[cursor * 2:(cursor + count) * 2].copy()
      pieces.append(((rows, cols), int(count), values, coordinates))
    cursor += count
  return pieces


def assemble_batched(minimizer, groups, assembled, gradient, batch_size):
  # Assemble ordinary scene energies once, then add each MPM material Hessian
  # in bounded particle batches. The matrix API sums duplicate sparse blocks.
  import pycuda.gpuarray as gpuarray
  base = minimizer.computeNumericValue()
  gradient.value[:] = base.gradient.value
  assembled.diagonal[:] = base.diagonal
  assembled.diagonal_blocks[:] = base.diagonal_blocks
  for name in ("block_dimensions", "blocks_flattened", "blocks_start_indices", "block_positions", "block_counts"):
    setattr(assembled, name, getattr(base, name))
  pieces = copy_block_part(base)
  for group in groups:
    try:
      for first in range(0, group["count"], batch_size):
        bind_particle_range(group, first, min(first + batch_size, group["count"]))
        for H in group["hessians"]:
          if H is group.get("material_hessian"):
            H.compute(coordinate_cache_key=(first, min(first + batch_size, group["count"])))
          else:
            H.compute()
          gradient.value[:] += H.gradient.value
          assembled.diagonal[:] += H.diagonal
          assembled.diagonal_blocks[:] += H.diagonal_blocks
          material = group.get("material_hessian")
          if material is None:
            pieces.extend(copy_block_part(H))
          elif H is not material:
            add_particle_wall_blocks(group, material, H)
        if group.get("material_hessian") is not None:
          pieces.extend(copy_block_part(group["material_hessian"]))
    finally:
      bind_particle_range(group)
  pieces.sort(key=lambda item: item[0])
  dims, counts, starts, values, positions = [], [], [], [], []
  start = 0
  for shape, count, value, coordinate in pieces:
    if not dims or tuple(dims[-2:]) != shape:
      dims.extend(shape)
      counts.append(0)
      starts.append(start)
    counts[-1] += count
    start += value.size
    values.append(value)
    positions.append(coordinate)
  assembled.block_dimensions_dynamic = dims
  assembled.block_counts_dynamic = counts
  assembled.blocks_start_indices_dynamic = starts
  assembled.blocks_flattened_dynamic = gpuarray.concatenate(values) if values else gpuarray.empty(0, np.float64)
  assembled.block_positions_dynamic = gpuarray.concatenate(positions) if positions else gpuarray.empty(0, np.uint32)
  return assembled


def add_particle_wall_blocks(group, material, walls):
  # Numerical accumulation only. All derivatives, projection, coordinates and
  # per-energy scatter were computed by YASPS. The material's full stencil
  # includes every active wall pair, even where a material value is zero.
  count = sum(walls.block_counts_dynamic)
  if not count:
    return
  dimensions = list(material.block_dimensions_dynamic)
  if dimensions not in ([1, 1], [3, 3]) or list(walls.block_dimensions_dynamic) != dimensions:
    raise ValueError("Particle wall accumulation expects matching scalar or vector grid blocks")
  group["add_wall_blocks"](walls.block_positions_dynamic, walls.blocks_flattened_dynamic, np.uint32(count), material.block_positions_dynamic, material.blocks_flattened_dynamic, np.uint32(sum(material.block_counts_dynamic)), np.uint32(dimensions[0] * dimensions[1]), block=(128, 1, 1), grid=((count + 127) // 128, 1, 1))


def full_energy(s, groups):
  # Particle material energies are deliberately outside the scene minimizer
  # so include their values explicitly in Armijo comparisons.
  import pycuda.gpuarray as gpuarray
  # Refilter at EVERY trial position, including particles that entered a wall
  # barrier since the Hessian was assembled. A zero active set contributes zero.
  for group in groups:
    update_particle_walls(group)
  return float(s.computeTotalEnergy() + sum(float(gpuarray.sum(energy.compute().value).get()) for group in groups for energy in group["energies"] if energy.correspondance.numInstances > 0))


def update_contacts(ccd, records, vertex_weights, transfer_module):
  import pycuda.gpuarray as gpuarray
  for record, pairs, count in zip(records, [ccd.pp, ccd.pe, ccd.pt, ccd.ee], ccd.separated_counts):
    record["primitive"].updateNumInstances(count)
    if count:
      record["connection"].updateConnectivity(pairs[:count * record["arity"]])
      weight = gpuarray.empty(count, np.float64)
      transfer_module.get_function("contact_weights")(pairs, vertex_weights, weight, np.int32(record["arity"]), np.int32(count), block=(128, 1, 1), grid=((count + 127) // 128, 1, 1))
      record["weight"].updateValue(weight)
