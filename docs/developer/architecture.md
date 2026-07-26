# Architecture

YASPS separates a backend-independent symbolic model from backend-specific
evaluation and sparse numerical work. Understanding that boundary is useful
when adding an operator, debugging generated code, or extending the Metal
backend.

## Layer map

```text
Python user program
  scene → mesh → primitive → attribute/connectivity/union
                       │
                       ▼
             symbolic attribute DAG
           shapes, lineage, operations
                       │
              symbolic derivatives
      gradient indices, Jacobians, Hessians
                       │
        sparse coordinate compression
                       │
                       ▼
       backend-specific numerical execution
       ├── CUDA: generated CUDA/Eigen kernels
       └── Metal: MLX + custom Metal kernels
                       │
                       ▼
       block preconditioner + PCG solution
```

The scene and expression graph should not contain CUDA- or Metal-specific
user-facing concepts. Backend selection happens below that interface.

## Shared interface and specialization points

High-level dispatch stays in the existing logical implementation files. This
keeps the Metal specialization next to the CUDA path for navigation:

- `attribute.pyx` selects expression evaluation;
- `gradientIndicesKernel.pyx` selects derivative-index generation;
- `coordinateCompressionKernel.pyx` selects sparse coordinate compression;
- `hessian.pyx` selects numerical assembly;
- `solver.pyx` selects sparse solve operations; and
- `minimizer.pyx` coordinates differentiation, assembly, and solving.

These files mark Metal branches with `METAL SPECIALIZATION` comments.

Low-level Metal implementation belongs in `yasps/metal/`, which acts as the
dedicated Metal numerical library:

- `evaluator.py` evaluates symbolic attribute graphs;
- `linalg.py` provides small-matrix eigendecomposition and SPD projection;
- `assembly.py` builds sparse gradient and Hessian values;
- `sparse.py` provides sparse matrix-vector products, block inverses, and PCG;
  and
- `ccd.py` provides Metal collision detection and continuous collision
  detection.

This layout avoids duplicating the scene API while keeping shader-oriented code
out of large Cython interface modules.

## Backend selection

`yasps.backend` is the common runtime facade. It reads `YASPS_BACKEND`:

| Value | Behavior |
| --- | --- |
| `auto` or unset | Probe usable CUDA first, then select Metal on Apple silicon |
| `cuda` | Require a working PyCUDA context |
| `metal` | Require Apple silicon and the MLX Metal runtime |

The facade exposes the small `gpuarray` surface used by the Cython modules.
On CUDA this is PyCUDA's implementation. On Metal, the compatibility wrapper
keeps live host-visible arrays while numerical kernels execute through MLX or
custom Metal code.

Code that merely needs an array, synchronization, or backend query should
import from `yasps.backend`, not directly from PyCUDA.

## Symbolic graph

An attribute node records:

- its row and column shape;
- the operation that produced it;
- parent operands;
- its owning primitive or mesh;
- data lineage through connectivities and unions; and
- whether it is mutable stored data, a constant, or a computed value.

Python operators and methods construct nodes; they do not immediately execute
the operation. `compute()` materializes a graph. Naming a computed attribute
attaches that graph to a primitive and gives later code a stable lookup point.

Shape information is part of the graph. Operations such as matrix
multiplication, transpose, determinant, inverse, reduction, indexing, and
stacking determine their output shape when the expression is constructed.

## Differentiation

`scene.addMinimizeTarget()` establishes the independent attributes and their
global order. For every registered scalar energy, YASPS symbolically
differentiates the graph with respect to those targets.

Topology lineage is as important as algebraic lineage. If a tetrahedron energy
uses positions gathered through a tetrahedron-to-vertex connectivity, the
derivative generator maps each local derivative block back to the global
vertex degrees of freedom.

The sparse pipeline is:

1. generate raw gradient and Hessian coordinates from the symbolic derivative
   graph and connectivities;
2. merge duplicate coordinates;
3. evaluate derivative expressions for every active energy instance;
4. assemble values into the compressed structure;
5. project local Hessian blocks when requested; and
6. solve \(H\Delta x=g\).

The result returned to Python is split in exactly the minimization-target order.

## Hessian projection and eigendecomposition

Energy registration chooses one projection method:

| Method | Local eigenvalue treatment |
| ---: | --- |
| `-1` | Skip projection |
| `0` | Keep eigenvalues unchanged |
| `1` | Replace each eigenvalue with its absolute value |
| `2` | Clamp negative eigenvalues to zero |

The CUDA path uses the existing generated Eigen-based implementation. The
Metal path uses a custom symmetric Jacobi eigensolver for the small block sizes
used by YASPS (`1`, `2`, `3`, `4`, `6`, `9`, and `12`). Projection reconstructs
the block from its eigenvectors and modified eigenvalues.

Metal shaders do not provide general double-precision arithmetic, so this path
uses float32. Float64 inputs are converted at the backend boundary. Tests must
therefore use tolerances appropriate to float32 and include nonsymmetric
matrices when validating inverse or determinant derivatives.

## Sparse solve

The Metal sparse path does not form a dense global Hessian. It keeps compressed
block coordinates, evaluates values on the GPU, applies sparse matrix-vector
products, builds block-diagonal inverses, and runs preconditioned conjugate
gradient.

Dynamic energy instance counts change assembled values and active terms without
changing the target layout. The solution buffer remains persistent so Cython
views returned for each target observe updated solver results on later calls.

## Collision subsystem

The shared entry point is `examples/ccd/ccd.py`:

- CUDA selects the original generated-kernel implementation;
- Metal selects `yasps.metal.ccd.MetalCCD`.

The Metal implementation builds Morton-ordered balanced AABB trees and traverses
them with custom kernels. It performs:

- discrete broad-phase candidate generation;
- robust point/edge/triangle feature classification;
- separated PP, PE, PT, and EE contact output;
- continuous collision candidate generation; and
- conservative advancement for a collision-free step size.

Mesh topology is supplied from host arrays because it changes rarely. Bounding
boxes, traversal, and numerical feature tests execute on the GPU.

Candidate output has an explicit capacity. Overflow is observable and callers
must retry with a larger capacity or fail; losing candidates silently would
invalidate contact simulation.

## Generated CUDA code

The CUDA backend still uses expression-specific generated kernels and Eigen
headers. Backend work should preserve that path. A shared Cython file may
dispatch to Metal, but it must not remove the existing CUDA code-generation
logic or change a CUDA example merely to make a Metal example run.

The separately named Metal dropping driver exists because the original driver
is a research artifact with CUDA-oriented imports and launch assumptions. Both
drivers build the same categories of YASPS objects.

## Adding a symbolic operator

An operator is complete only when all relevant layers agree:

1. define its graph node, shape rules, and Python entry point;
2. define symbolic first and second derivatives;
3. add CUDA generation or evaluation;
4. add Metal evaluation;
5. test scalar, vector, matrix, and batched-instance cases as applicable;
6. test derivatives with finite differences; and
7. document public support in the operator table.

Do not expose an operation merely because an internal opcode exists. A public
operator also needs a stable construction method and backend implementations.

## Validation strategy

Use tests in increasing scope:

1. backend array and live-view behavior;
2. individual expression operators;
3. finite-difference gradients and Hessians;
4. small-matrix eigendecomposition and projection;
5. sparse assembly against dense reference matrices;
6. PCG against a known solution;
7. isolated PP, PE, PT, and EE CCD cases;
8. a tiny complete scene;
9. one dropping frame;
10. a run through first impact and subsequent friction frames.

A pre-impact run validates gravity, elasticity, and solving, but not the
collision implementation. Record mesh sizes, contact counts, overflow retries,
solver convergence, and accepted energy changes for meaningful performance and
correctness comparisons.
