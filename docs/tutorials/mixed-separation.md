---
title: Mixed bodies with separated assembly
description: Build and solve the five-bunny container example across soft, affine, static, and dynamic contact parameterizations.
permalink: /tutorials/mixed-separation/
---

<p class="eyebrow">Complete frontend walkthrough</p>

# Mixed bodies, one collision objective

<p class="lead">This tutorial follows the real <code>dropping_in_container_mixed_separation</code> program from mesh loading to velocity update. With five bunnies, four use free vertex degrees of freedom, one uses an affine matrix and translation, and all five collide with the same static container through one symbolic frontend.</p>

<div class="hero-actions">
  <a class="button" href="https://github.com/txstc55/yasps/blob/main/examples/dropping_in_container_mixed_separation/dropping_in_container.py">Open the complete program</a>
  <a class="button secondary" href="https://github.com/txstc55/yasps/blob/main/examples/dropping_in_container_mixed_separation/helpers.py">Open the energy helpers</a>
</div>

## What the example demonstrates

<ol class="pipeline">
  <li>Declare two parameterizations</li>
  <li>Gather with JOIN</li>
  <li>Combine with UNION</li>
  <li>Generate separated contact assembly</li>
  <li>Solve, CCD, and accept</li>
</ol>

The frontend does not branch into a soft-contact implementation and an affine-contact implementation. It gives every collision vertex a `position` attribute, unions those attributes, and writes each contact stencil once. Differentiation follows the symbolic route back to the correct degrees of freedom.

> The code blocks below are focused excerpts that read in sequence. The linked production file contains the complete NumPy uploads, timing instrumentation, CCD allocation, and optional rendering setup.

Run the original from its directory so the relative data and CCD imports resolve:

```bash
cd examples/dropping_in_container_mixed_separation
python dropping_in_container.py --num-bunnies 5
```

The script imports PyVista even when `--save-obj` is omitted, so its example environment needs PyVista in addition to the core YASPS and CUDA dependencies. The `--save-obj True` path opens an interactive PyVista view; the OBJ-writing lines at the end of the current script remain commented out.

<div class="source-map">
  <a href="https://github.com/txstc55/yasps/blob/main/examples/dropping_in_container_mixed_separation/dropping_in_container.py#L157-L230">Soft frontend · lines 157–230</a>
  <a href="https://github.com/txstc55/yasps/blob/main/examples/dropping_in_container_mixed_separation/dropping_in_container.py#L235-L322">Affine frontend · lines 235–322</a>
  <a href="https://github.com/txstc55/yasps/blob/main/examples/dropping_in_container_mixed_separation/dropping_in_container.py#L329-L357">UNION + contact · lines 329–357</a>
  <a href="https://github.com/txstc55/yasps/blob/main/examples/dropping_in_container_mixed_separation/dropping_in_container.py#L366-L405">Energy registry · lines 366–405</a>
  <a href="https://github.com/txstc55/yasps/blob/main/examples/dropping_in_container_mixed_separation/dropping_in_container.py#L486-L628">Frame/Newton loop · lines 486–628</a>
  <a href="https://github.com/txstc55/yasps/blob/main/examples/dropping_in_container_mixed_separation/helpers.py#L129-L208">Symbolic energies · helper lines 129–208</a>
</div>

<section class="step" data-step="01" markdown="1">

## Establish the scene-wide parameters

The example loads one tetrahedral bunny, builds five translated copies, and splits them into four soft bodies and one affine body. It then creates scalar constants shared by every energy:

```python
NUM_BUNNIES = 5
NUM_AFFINE_BUNNIES = 1

DT_VALUE = 0.01
DHAT_VALUE = 1e-6
KAPPA_VALUE = 10000.0

world = scene("scene0")

dt = world.addConstant("dt")
dt.updateValue(DT_VALUE)

dhat = world.addConstant("dhat")
dhat.updateValue(DHAT_VALUE)

kappa = world.addConstant("kappa")
kappa.updateValue(KAPPA_VALUE)
```

`addConstant` means “derivative is zero,” not “value can never change.” These scene values remain GPU data that generated kernels can read.

The mesh files, surface extraction, copied indices, and material constants are ordinary Python/NumPy setup. YASPS starts where the program declares the scene hierarchy and symbolic attributes.

</section>

<section class="step" data-step="02" markdown="1">

## Give the soft bodies vertex DOFs

The soft parameterization stores one `3 × 1` target per vertex. Rest position, previous position, velocity, and mass are mutable constants because the solver must not differentiate with respect to them:

```python
soft_mesh = world.addMesh("bunnies_soft")
mu_soft = soft_mesh.addConstant("mu")
lam_soft = soft_mesh.addConstant("lambda")

soft_vertices = soft_mesh.addPrimitive(
    "vertices_soft",
    numInstances=4 * NUM_BUNNY_VERTICES,
)

x_soft = soft_vertices.addAttribute("position", rows=3, cols=1)
x_rest = soft_vertices.addConstant("rest_position", rows=3, cols=1)
x_last = soft_vertices.addConstant("last_position", rows=3, cols=1)
v_soft = soft_vertices.addConstant("velocity", rows=3, cols=1)
mass_soft = soft_vertices.addConstant("mass", rows=1, cols=1)
```

The actual program uploads flattened float64 NumPy arrays with `updateValue`. Only `x_soft` will appear in the final minimization-target list.

Tetrahedra need four vertex positions per instance. The connectivity makes that route explicit:

```python
soft_tets = soft_mesh.addPrimitive(
    "tets_soft",
    numInstances=4 * NUM_BUNNY_TETS,
)

tet2vertex = soft_tets.addConnectivity(
    "tets_softs2_vertices",
    soft_vertices,
    bunny_tet_indices_soft,
    4,
)

tet_x = soft_tets.addAttribute(
    "positions",
    through=tet2vertex,
    source=x_soft,
)
tet_x_rest = soft_tets.addAttribute(
    "rest_positions",
    through=tet2vertex,
    source=x_rest,
)
```

Each gathered attribute is a `4 × 3` per-tetrahedron matrix. No full gathered array is created here: JOIN contributes index traversal to the eventual generated kernel and to the reverse derivative path.

</section>

<section class="step" data-step="03" markdown="1">

## Build a reusable deformation expression

The example forms rest and current edge matrices from the four gathered rows:

```python
def edge_matrix(tet_positions):
    p0 = tet_positions.row(0)
    e0 = tet_positions.row(1) - p0
    e1 = tet_positions.row(2) - p0
    e2 = tet_positions.row(3) - p0
    return attribute.to_array(
        [
            e0[0], e0[1], e0[2],
            e1[0], e1[1], e1[2],
            e2[0], e2[1], e2[2],
        ],
        rows=3,
        cols=3,
    )

TB = soft_tets.addAttribute("TB", computed_attribute=edge_matrix(tet_x_rest))
F = soft_tets.addAttribute("F", computed_attribute=edge_matrix(tet_x))
```

The production script moves those matrices through a one-to-one JOIN onto a `deformation_gradient` primitive and calls `resize(3, 3)`. That extra boundary lets the generated separation path treat each deformation-gradient instance as a compact local energy domain.

Both soft and affine bodies later call the same helper:

```python
def stable_neo_hookean_modified(F, TB, mu, lam, dt):
    inverse_basis = TB.transpose().inverse()
    volume = TB.transpose().determinant() / 6.0
    deformation = F.transpose() * inverse_basis
    J = deformation.determinant()
    Ic = (deformation.transpose() * deformation).trace()
    I3 = Ic + 1.0

    return volume * (
        0.5 * mu * (Ic - 3.0)
        - 0.5 * mu * I3.log()
        + 0.5 * lam
        * (J - (1.0 + 0.75 * mu / lam))
        * (J - (1.0 + 0.75 * mu / lam))
    ) * dt * dt
```

Every line constructs `attribute` nodes. Python does not materialize `deformation`, `J`, or the partial energy arrays.

</section>

<section class="step" data-step="04" markdown="1">

## Parameterize one bunny affinely

The affine body has 12 scalar DOFs: a `3 × 3` matrix `A` and a `3 × 1` translation `t`.

```python
affine_mesh = world.addMesh("bunnies_abd")
mu_affine = affine_mesh.addConstant("mu")
lam_affine = affine_mesh.addConstant("lambda")

affine_bodies = affine_mesh.addPrimitive(
    "affine_bodies",
    numInstances=1,
)

A = affine_bodies.addAttribute("affine_matrices", rows=3, cols=3)
t = affine_bodies.addAttribute("translations", rows=3, cols=1)
```

Affine vertices are not independent targets. A one-to-one JOIN broadcasts the owning body's parameters to every vertex, then a named computed attribute defines world position:

```python
affine_vertices = affine_mesh.addPrimitive(
    "vertices_abd",
    numInstances=NUM_BUNNY_VERTICES,
)
r = affine_vertices.addConstant("rest_position", rows=3, cols=1)
affine_x_last = affine_vertices.addConstant(
    "last_position",
    rows=3,
    cols=1,
)
v_affine = affine_vertices.addConstant("velocity", rows=3, cols=1)
mass_affine = affine_vertices.addConstant("mass", rows=1, cols=1)

vertex2body = affine_vertices.addConnectivity(
    "v_abd2_abd",
    affine_bodies,
    [[0] * NUM_BUNNY_VERTICES],
    1,
)

vertex_A = affine_vertices.addAttribute(
    "affine_matrix",
    through=vertex2body,
    source=A,
).resize(3, 3)

vertex_t = affine_vertices.addAttribute(
    "translation",
    through=vertex2body,
    source=t,
).resize(3, 1)

x_affine = affine_vertices.addAttribute(
    "position",
    computed_attribute=vertex_A * r + vertex_t,
)
```

The frontend now exposes the same `position` name and `3 × 1` shape on soft and affine vertices, but their derivative routes differ:

```text
soft position   → vertex position data
affine position → A · rest_position + t
```

Affine tetrahedra gather rest positions and body parameters, construct their current positions, and build `F` and `TB` exactly as the soft side did. The same stable Neo-Hookean helper therefore generates a soft-vertex Hessian on one route and an affine-parameter Hessian on the other.

The affine inertia energy also operates on computed vertex positions, so its derivatives accumulate into `A` and `t`. A separate `affine_energy(A)` penalizes `AᵀA − I`, keeping the affine transformation near orthogonal.

</section>

<section class="step" data-step="05" markdown="1">

## Present one position field to collision

The container contributes a third `position` attribute. It is numerical data, but it is deliberately absent from the minimization-target list, so it affects energy evaluation without receiving an update segment.

```python
container = world.addMesh("container")
container_vertices = container.addPrimitive(
    "vertices",
    numInstances=len(container_positions),
)
container_x = container_vertices.addAttribute("position", rows=3, cols=1)
container_x.updateValue(container_positions)
```

UNION stacks the three populations in a known order:

```python
collision_mesh = world.addMesh("collision_mesh")

collision_vertices = collision_mesh.addPrimitiveUnion(
    "vertices",
    [soft_vertices, affine_vertices, container_vertices],
)

collision_x = collision_vertices.addAttribute("position")
```

The union ordering must match the indices passed to CCD:

```text
[ all soft vertices ][ all affine vertices ][ container vertices ]
```

`collision_x` is one symbolic field. On a soft index it reads `x_soft`; on an affine index it evaluates `A · r + t`; on a container index it reads the container buffer. UNION preserves those routes rather than flattening them into disconnected values.

</section>

<section class="step" data-step="06" markdown="1">

## Define four dynamic contact stencil types

The candidate set changes during line search, but the symbolic energy shape does not. Create empty dynamic primitives once:

```python
pp = collision_mesh.addPrimitive("pp", 0, isDynamic=True)
pe = collision_mesh.addPrimitive("pe", 0, isDynamic=True)
pt = collision_mesh.addPrimitive("pt", 0, isDynamic=True)
ee = collision_mesh.addPrimitive("ee", 0, isDynamic=True)

pp2v = pp.addConnectivity("pp2v", collision_vertices, [], 2)
pe2v = pe.addConnectivity("pe2v", collision_vertices, [], 3)
pt2v = pt.addConnectivity("pt2v", collision_vertices, [], 4)
ee2v = ee.addConnectivity("ee2v", collision_vertices, [], 4)

pp_x = pp.addAttribute("positions", through=pp2v, source=collision_x)
pe_x = pe.addAttribute("positions", through=pe2v, source=collision_x)
pt_x = pt.addAttribute("positions", through=pt2v, source=collision_x)
ee_x = ee.addAttribute("positions", through=ee2v, source=collision_x)
```

The contact helper functions consume only a gathered matrix and scalar constants. For example:

```python
def point_point(position, dHat, kappa):
    p0 = position.row(0)
    p1 = position.row(1)
    distance_squared = (p1 - p0).dot(p1 - p0)
    normalized = distance_squared / dHat
    offset = distance_squared - dHat
    log_term = normalized.log()
    return kappa * offset * offset * log_term * log_term
```

Point–edge, point–triangle, and edge–edge use the same pattern with their own symbolic squared-distance formula. None of them asks whether a row came from a soft body, affine body, or container.

</section>

<section class="step" data-step="07" markdown="1">

## Register energy policies and turn on separation

Name every scalar energy on its owning primitive, then register its numerical policy:

```python
snh_softs = bdg.addAttribute(
    "snh_softs",
    computed_attribute=stable_neo_hookean_modified(
        bdg_F,
        bdg_TB,
        soft_mesh["mu"],
        soft_mesh["lambda"],
        dt,
    ),
)
snh_abds = bdg_abd.addAttribute(
    "snh_abds",
    computed_attribute=stable_neo_hookean_modified(
        bdg_F_abd,
        bdg_TB_abd,
        affine_mesh["mu"],
        affine_mesh["lambda"],
        dt,
    ),
)

inertia_softs = soft_vertices.addAttribute(
    "inertia_softs",
    computed_attribute=inertia(x_last, v_soft, dt, x_soft, mass_soft),
)
inertia_abds = affine_vertices.addAttribute(
    "inertia_abds",
    computed_attribute=inertia(
        affine_x_last,
        v_affine,
        dt,
        x_affine,
        mass_affine,
    ),
)
affine_constraint = affine_bodies.addAttribute(
    "affine_energy",
    computed_attribute=affine_energy(A),
)

point_point_e = pp.addAttribute(
    "point_point",
    computed_attribute=point_point(pp_x, dhat, kappa),
)
point_edge_e = pe.addAttribute(
    "point_edge",
    computed_attribute=point_edge(pe_x, dhat, kappa),
)
point_triangle_e = pt.addAttribute(
    "point_triangle",
    computed_attribute=point_triangle(pt_x, dhat, kappa),
)
edge_edge_e = ee.addAttribute(
    "edge_edge",
    computed_attribute=edge_edge(ee_x, dhat, kappa),
)

world.addEnergy(snh_softs, projection_method=1)
world.addEnergy(snh_abds, projection_method=1)

world.addEnergy(inertia_softs, projection_method=-1)
world.addEnergy(inertia_abds, projection_method=-1)

world.addEnergy(affine_constraint, projection_method=2)

for contact_energy in [point_point_e, point_edge_e, point_triangle_e, edge_edge_e]:
    world.addEnergy(
        contact_energy,
        dynamic_instances=True,
        projection_method=2,
        separate_hessian_jacobian=True,
    )
```

The policies are intentionally different:

| Term | Structure | Projection | Why |
| --- | --- | --- | --- |
| Soft/affine elasticity | Static | absolute eigenvalues (`1`) | Stabilize the local nonlinear Hessian |
| Inertia | Static | skipped (`-1`) | Already convex; avoid an unnecessary projection |
| Affine orthogonality | Static | clamp negative (`2`) | Stabilize the constraint term |
| PP/PE/PT/EE contact | Dynamic | clamp negative (`2`) | Candidate counts and sparse placements change |

`separate_hessian_jacobian=True` changes code generation, not the energy. Instead of expanding a large topology/UNION Jacobian into one monolithic symbolic Hessian expression, YASPS keeps the local Hessian and sparse outer Jacobian as separate generated stages, then reorders placement lookups for assembly. This is especially useful for contact paths that touch several heterogeneous target segments.

The static elastic and inertia terms reuse their sparse structure. The four contact terms refresh their coordinates whenever candidate counts change.

</section>

<section class="step" data-step="08" markdown="1">

## Declare the global solution layout

The target list is the contract between symbolic differentiation and the returned solve:

```python
targets = [x_soft, A, t]
world.addMinimizeTarget(targets)

directions = world.minimizeEnergy(tolerance=1e-4)
dx_soft, dA, dt_affine = directions
```

The flattened global vector is ordered exactly as the target list:

| Segment | Per-instance shape | Instances | Meaning |
| --- | --- | --- | --- |
| `directions[0]` | `3 × 1` | all soft vertices | Free-vertex direction |
| `directions[1]` | `3 × 3` | one affine bunny | Affine-matrix direction |
| `directions[2]` | `3 × 1` | one affine bunny | Translation direction |

Container positions do not appear. Affine vertex positions do not appear either because they are computed expressions, not `DATA` targets.

YASPS solves `H Δx = g`. The application applies the negative direction:

```python
x_soft.updateValue(x_soft.value - dx_soft, deepCopy=True)
A.updateValue(A.value - dA, deepCopy=True)
t.updateValue(t.value - dt_affine, deepCopy=True)
```

</section>

<section class="step" data-step="09" markdown="1">

## Feed CCD back into dynamic topology

Collision detection lives outside the symbolic package. The example initializes its CCD helper with surface points, edges, triangles, and a mesh label for each vertex. Those arrays use the exact UNION ordering from step 5.

During backtracking, CCD produces four flat GPU index buffers. Update both the instance count and connectivity:

```python
pp_count, pe_count, pt_count, ee_count = ccd.separated_counts

pp.updateNumInstances(pp_count)
pe.updateNumInstances(pe_count)
pt.updateNumInstances(pt_count)
ee.updateNumInstances(ee_count)

if pp_count:
    pp2v.updateConnectivity(ccd.pp[: 2 * pp_count])
if pe_count:
    pe2v.updateConnectivity(ccd.pe[: 3 * pe_count])
if pt_count:
    pt2v.updateConnectivity(ccd.pt[: 4 * pt_count])
if ee_count:
    ee2v.updateConnectivity(ccd.ee[: 4 * ee_count])
```

The symbolic barrier expressions remain untouched. On the next assembly, dynamic index kernels recompute which global blocks these active stencil instances reach.

</section>

<section class="step" data-step="10" markdown="1">

## Own the Newton and timestep loop

One frame in the original program has three levels of responsibility:

```python
for frame in range(200):
    # State history for inertia.
    x_last.updateValue(x_soft.value, deepCopy=True)
    affine_x_last.updateValue(x_affine.compute().value, deepCopy=True)

    while True:
        # YASPS: assemble H/g and solve H Δx = g.
        dx_soft, dA, dt_affine = world.minimizeEnergy(tolerance=1e-4)
        energy_before = world.computeTotalEnergy()

        # Application: preserve the accepted state.
        x0 = x_soft.value.copy()
        A0 = A.value.copy()
        t0 = t.value.copy()
        union_x0 = collision_x.compute().value.copy()

        # Application: test a full step and ask CCD for a safe bound.
        x_soft.updateValue(x0 - dx_soft, deepCopy=True)
        A.updateValue(A0 - dA, deepCopy=True)
        t.updateValue(t0 - dt_affine, deepCopy=True)
        direction_world = union_x0 - collision_x.compute().value
        ccd.ccd(union_x0, DHAT_VALUE, direction_world, 0.5)
        alpha = ccd.compute_largest_step_size(
            0.5,
            union_x0,
            direction_world,
        )

        # Application + YASPS dynamic frontend: backtrack and refresh contacts.
        for _ in range(8):
            x_soft.updateValue(x0 - alpha * dx_soft, deepCopy=True)
            A.updateValue(A0 - alpha * dA, deepCopy=True)
            t.updateValue(t0 - alpha * dt_affine, deepCopy=True)

            ccd.cd(collision_x.compute().value, DHAT_VALUE)
            update_dynamic_contact_primitives()

            if world.computeTotalEnergy() <= energy_before:
                break
            alpha *= 0.5

        if max_world_velocity(direction_world, DT_VALUE) < 1e-2:
            break

    # State integration after the accepted frame.
    v_soft.updateValue(
        (x_soft.value - x_last.value) / DT_VALUE,
        deepCopy=True,
    )
    v_affine.updateValue(
        (x_affine.compute().value - affine_x_last.value) / DT_VALUE,
        deepCopy=True,
    )
```

The excerpt names the same operations as the production loop while compressing its timing and rendering code. The boundary is deliberate:

- YASPS constructs derivatives, sparse placements, CUDA kernels, numerical assembly, preconditioning, and PCG.
- The application owns collision detection, CCD step limits, backtracking, convergence, state history, velocity updates, and rendering.

`computeTotalEnergy()` returns a Python scalar and therefore synchronizes with the GPU. The example needs that value for backtracking. Ordinary symbolic subexpressions stay fused and device-side.

</section>

## The frontend in one map

| Frontend concept | Where it appears in this example | Read next |
| --- | --- | --- |
| Scene/mesh/primitive hierarchy | Soft, affine, container, and collision meshes | [Mental model]({{ '/concepts/' | relative_url }}) |
| Data versus constant leaves | Position targets versus rest/history/material state | [Attributes]({{ '/attributes/' | relative_url }}) |
| Symbolic matrix syntax | Edge matrices, Neo-Hookean energy, affine constraint, barriers | [Attributes]({{ '/attributes/' | relative_url }}) |
| JOIN | Tet-to-vertex, vertex-to-body, deformation-gradient boundaries, contact stencils | [Connectivity and JOIN]({{ '/join/' | relative_url }}) |
| UNION | One collision position field over three parameterizations | [Primitive unions]({{ '/union/' | relative_url }}) |
| Dynamic topology | PP/PE/PT/EE candidate counts and connectivity | [Dynamic topology]({{ '/dynamic-scenes/' | relative_url }}) |
| Energy policies | Projection, dynamic instances, separated Hessian/Jacobian | [Energies and minimization]({{ '/optimization/' | relative_url }}) |
| Global target layout | Soft positions, affine matrices, translations | [Direct minimizer use]({{ '/advanced/minimizer/' | relative_url }}) |
| Generated sparse solve | Target paths → indices → assembly → PCG | [How YASPS executes]({{ '/architecture/' | relative_url }}) |

Once this flow is clear, the larger `brazil_nuts`, cage, cloth, and teaser examples are variations in model construction—not different frontend rules.
