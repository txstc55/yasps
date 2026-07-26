# Dropping in a container

`examples/dropping_in_container/dropping_in_container_metal.py` is a separate,
headless-friendly version of the full dropping experiment. It preserves the
original CUDA driver and exercises the shared YASPS interface through the
selected backend.

It is the best end-to-end example of how topology, symbolic energy
construction, automatic differentiation, dynamic contact, and iterative
solving fit together.

## Run it

From the repository root:

```bash
python examples/dropping_in_container/dropping_in_container_metal.py \
  --steps 60 \
  --max-newton 20
```

On Apple silicon, automatic backend selection chooses Metal. Select it
explicitly while debugging:

```bash
YASPS_BACKEND=metal \
python examples/dropping_in_container/dropping_in_container_metal.py \
  --steps 1 \
  --max-newton 2 \
  --solver-iterations 500
```

Use `--save-meshes` to write an OBJ surface for each completed frame.

## Render saved frames

Install the optional PyVista renderer:

```bash
python -m pip install -e './yasps[render]'
```

Then turn the accepted OBJ frames into a camera-stable MP4 and final PNG:

```bash
python examples/dropping_in_container/render_metal.py \
  --input-directory examples/dropping_in_container/meshes_metal_5_bunnies \
  --num-bunnies 5 \
  --video examples/dropping_in_container/artifacts/dropping_metal_5_bunnies.mp4 \
  --screenshot examples/dropping_in_container/artifacts/dropping_metal_5_bunnies.png
```

Rendering is deliberately separate from simulation. A VTK window or video
encoder therefore cannot interfere with Metal synchronization, solver timing,
or a long headless run. The script uses a fixed camera, a translucent
container, stable per-bunny colors, topology validation, and off-screen
anti-aliasing.

## Scene layout

The example builds three meshes:

```text
scene
├── soft_mesh
│   ├── vertices       current and historical state
│   └── tetrahedra     elastic elements
├── fixed_mesh
│   └── vertices       container geometry
└── collision_mesh
    ├── vertices       union of soft and fixed vertices
    ├── pp/pe/pt/ee    dynamic barrier interactions
    └── *_friction     dynamic lagged-friction interactions
```

The collision vertex union exposes child attributes with:

```python
collision_vertices = collision_mesh.addPrimitiveUnion(
  "vertices", [soft_vertices, fixed_vertices]
)
collision_vertices.addAttribute("position")
collision_vertices.addAttribute("last_position")
```

The order of the children determines the global collision-vertex numbering.
The triangle, edge, and surface-vertex arrays use the same ordering.

## Elasticity

Each tetrahedron gathers four current and four rest positions through a
connectivity:

```python
tet_to_vertex = tetrahedra.addConnectivity(
  "tet2vertex", soft_vertices, tetrahedron_indices, 4
)
tet_position = tetrahedra.addAttribute(
  "position", through=tet_to_vertex, source=position
)
tet_rest_position = tetrahedra.addAttribute(
  "rest_position", through=tet_to_vertex, source=rest_position
)
```

The helper constructs a stable Neo-Hookean density from the deformation
gradient, rest volume, and Lamé parameters. The final expression is scalar for
each tetrahedron and is registered with Hessian projection method `1`.

The symbolic determinant and inverse therefore participate in both first and
second derivatives. On Metal these 3×3 operations are evaluated in float32.

## Inertia and gravity

The vertex inertia term uses the usual implicit-Euler prediction:

```text
x_predicted = x_last + dt * velocity + dt² * gravity
E_inertia   = 0.5 * mass / dt² * ||x - x_predicted||²
```

Only the current soft-vertex position is a minimization target. Rest
positions, mass, time step, velocity, and historical state are constants:
they may change between frames, but are excluded from differentiation.

## Barrier contact

The broad phase receives:

- the complete collision-vertex position array;
- surface triangles;
- unique surface edges; and
- surface-vertex indices.

It returns four feature-classified pair arrays: point–point, point–edge,
point–triangle, and edge–edge. Those arrays become the connectivity of four
dynamic primitives:

```python
primitive.updateNumInstances(pair_count)
connectivity.updateConnectivity(pair_indices)
```

The barrier energy expressions were attached when the scene was built. Changing
the active count and indices changes which instances are evaluated without
rebuilding the expression graph.

For Metal, the broad phase uses Morton-ordered balanced AABB trees and GPU
tree traversal. A capacity flag detects when a query produced more candidates
than its output allocation. The driver retries with a larger capacity up to
`--max-candidates-per-query` rather than silently dropping contacts.

## Friction

At the beginning of each frame, the current separated contact set becomes the
lagged friction set. Each friction primitive gathers both current and previous
positions. Named computed attributes store:

- closest-point coordinates;
- a tangent basis; and
- the lagged normal-force magnitude.

The friction energy then uses those named results. Naming the intermediate
attributes makes the graph easier to inspect and avoids duplicating a long
subexpression in user code.

## Newton and line search

Every frame follows this loop:

1. Update lagged friction pairs from the prior accepted state.
2. Evaluate the current total energy.
3. Call `scene.minimizeEnergy()` to solve \(H\Delta x=g\).
4. Test `x_trial = x - alpha * delta`.
5. Run continuous collision detection from the accepted position to the trial
   position.
6. Limit the trial by the collision-free step size.
7. Recompute discrete contacts and total energy at the trial position.
8. Accept an energy-reducing step or halve `alpha`.

If no line-search trial is accepted, the example restores the last accepted
position. This matters because `updateValue()` mutates the live scene state.

The float32 Metal path permits a tiny scale-relative energy slack in the
acceptance test. It covers rounding at large total energies without accepting
a materially uphill step.

## State update

After the Newton loop, velocity is updated from the frame displacement:

```python
velocity.updateValue((position.value - last_position.value) / dt)
last_position.updateValue(position.value)
```

The example keeps these arrays as persistent attributes; updating them does not
rebuild or redifferentiate the energy graph.

## Command-line controls

| Argument | Default | Purpose |
| --- | ---: | --- |
| `--num-bunnies` | `1` | Number of deformable copies |
| `--steps` | `500` | Number of simulation frames |
| `--max-newton` | `100` | Newton iteration limit per frame |
| `--solver-iterations` | `20000` | PCG iteration limit |
| `--max-line-search` | `12` | Backtracking limit |
| `--max-newton-displacement` | `0.1` | Infinity-norm cap before CCD |
| `--candidates-per-query` | `128` | Initial Metal broad-phase capacity |
| `--max-candidates-per-query` | `8192` | Maximum retry capacity |
| `--save-meshes` | off | Save accepted soft surfaces |
| `--output-directory` | `meshes_metal` | Override output location |

Use fewer Newton and solver iterations for smoke tests, not as a replacement
for convergence checks in production experiments.

## What this validates

A run that reaches contact exercises substantially more than scene setup:

- symbolic matrix inverse and determinant;
- stable Neo-Hookean gradients and Hessians;
- sparse coordinate compression and assembly;
- block-diagonal inversion and PCG;
- all four discrete contact types;
- continuous collision detection;
- dynamic primitive resizing;
- barrier and friction differentiation;
- repeated live updates; and
- accepted-state restoration during line search.

That makes impact, rather than a pre-impact gravity frame, the meaningful
acceptance threshold for backend work.
