# Discrete-adjoint bunny drop

`adjoint_bunny.py` is a complete implicit inverse-simulation example on the
repository's full TetGen bunny:

- `19,193` vertices from `examples/data/bunny.node`
- `79,935` tetrahedra from `examples/data/bunny.ele`
- `20,832` extracted surface triangles
- `31,248` extracted surface edges

The mesh is normalized to a height of `0.35`. The default drop starts at the
visibly off-origin translation `(0.18, 0.45, 0.08)`, uses a large rendered
floor, and advances with `dt=0.01`.

## Objective and design fields

All three optimization modes minimize the same final-frame position loss:

```text
L = 1/N sum(i=1..N) ||x[T,i] - (X_rest[i] + target_position)||².
```

There is no synthetic target trajectory and no target mass, Young's modulus,
or Poisson ratio. The default target keeps the drop's horizontal offset and
places the undeformed bunny's bottom at the contact activation height:

```text
target_position = (0.18, sqrt(d_hat), 0.08).
```

Use `--target-position X Y Z` to select a different target configuration.
Keeping the default horizontal offset is important: a horizontal,
frictionless floor and internal elastic forces cannot change the bunny's
rigid horizontal translation.

The independently optimized controls are:

- `young`: one Young value and one Poisson value per tetrahedron; both fields
  are updated together.
- `mass`: one mass value per vertex. The default `8e-5` per vertex gives a
  total mass of about `1.54`.
- `initial-position`: one three-component translation shared by the initial
  vertex positions.

## Forward and backward systems

The example does not call `scene.addEnergy`, `scene.addMinimizeTarget`, or
`scene.minimizeEnergy`. Its forward Newton and backward adjoint operations use
the differentiated matrices directly:

```text
Phi[k+1](x; x[k], v[k], theta) =
    inertia(x, x[k] + h v[k] + h² gravity, vertex_mass)
  + h² elasticity(x, young, poisson)
  + floor_contact(x)
  + self_contact(x)

x[k+1] = argmin Phi[k+1]
v[k+1] = (x[k+1] - x[k]) / h.
```

The reverse pass uses the complete position/velocity state:

```text
A   = d² Phi / d(position)²
B_x = d² Phi / d(position) d(previous_position)
B_v = d² Phi / d(position) d(previous_velocity)
C   = d² Phi / d(position) d(design)

q[k+1] = lambda_x[k+1] + lambda_v[k+1] / h
Aᵀ mu = q[k+1]

design_gradient += -Cᵀ mu
lambda_x[k] = dL/dx[k] - B_xᵀ mu - lambda_v[k+1] / h
lambda_v[k] = -B_vᵀ mu.
```

Every rectangular second-order matrix retains the same strict chain rule as
the Hessian:

```text
J_rowᵀ H_inner J_column + H_recursive.
```

The row and column outer Jacobians remain distinct, the inner Hessian remains
explicit, and the recursive second-order term is added separately. A
constant outside the differentiation targets contributes zero.

Forward Newton uses a PSD globalization and GPU CG. The backward pass rebuilds
the exact converged-stationarity Hessian. Exact geometric contact Hessians can
be indefinite; if CG detects non-positive curvature, this example exports the
same symmetric block matrix and solves it with MINRES. This fallback changes
only the linear solver, not the Hessian or mixed-Jacobian chain rule.

The defaults relevant to impact convergence are:

```text
dt                         = 0.01
d_hat                      = 1e-6
max_newton_iterations      = 300
newton_tolerance           = 1e-12
contact_newton_tolerance   = 2e-7
```

The looser contact threshold handles closest-feature switching at impact; the
smooth frames retain the stricter tolerance needed for multi-step adjoint
accuracy.

## Running and rendering

Run all three modes:

```bash
PYTHONPATH=yasps python examples/inverse_simulation/adjoint_bunny.py
```

Or use a convenience entry point:

```bash
PYTHONPATH=yasps python examples/inverse_simulation/optimize_bunny_young.py
PYTHONPATH=yasps python examples/inverse_simulation/optimize_bunny_initial_position.py
PYTHONPATH=yasps python examples/inverse_simulation/optimize_bunny_mass.py
```

The measured large-mesh comparison uses 31 frames. This reaches the first
floor impact and activates thousands of dynamic self-contact terms without
entering the much more expensive severe-compression frames:

```bash
PYTHONPATH=yasps python examples/inverse_simulation/adjoint_bunny.py \
  --parameter all \
  --frames 31 \
  --optimization-steps 6 \
  --video-directory examples/inverse_simulation/videos_large \
  --json-output examples/inverse_simulation/videos_large/results.json
```

The example uses NumPy, SciPy, PyCUDA, and PyVista; video encoding additionally
requires `ffmpeg`.

For each design, the renderer keeps all 32 side-by-side PNGs under
`videos_large/<design>/frames/` and writes a 32-frame, 1280x640 H.264 video
to `videos_large/<design>_before_after.mp4`. It does not write OBJ
intermediates.

Add `--check-gradient` and use one optimization step for a central
finite-difference comparison:

```bash
PYTHONPATH=yasps python examples/inverse_simulation/adjoint_bunny.py \
  --parameter all \
  --frames 31 \
  --optimization-steps 1 \
  --check-gradient
```

## Measured validation

The full 31-frame, six-step run produced:

| Design | Initial loss | Final loss | Reduction | Final design summary |
| --- | ---: | ---: | ---: | --- |
| Young + Poisson per element | 1.03861e-3 | 8.28047e-4 | 20.27% | means: Young `3680.07`, Poisson `0.333816` |
| Initial translation | 1.03861e-3 | 5.77423e-9 | 99.9994% | `(0.180000, 0.487500, 0.080001)` |
| Mass per vertex | 1.03861e-3 | 8.97285e-4 | 13.61% | mean `4.87524e-5`, total `0.935706` |

The initial trajectory reached `3,521` self-contact pairs. The optimized
material, initial-position, and mass trajectories peaked at `1,691`, `26`,
and `1,147` pairs, respectively. Their maximum stationarity residuals were
`1.03e-7`, `1.41e-7`, and `8.26e-8`, all below the contact threshold.

Directional adjoint checks through the first impact measured relative errors
of:

- joint Young/Poisson: `4.33e-3`
- initial translation: `2.71e-4`
- per-vertex mass: `4.00e-3`

A separate 20-frame smooth mass check reached `6.15e-5` relative error with
the `1e-12` smooth Newton tolerance.
