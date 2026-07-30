# Discrete-adjoint bunny drop

`adjoint_bunny.py` is a complete implicit inverse-simulation example on the
repository's full TetGen bunny:

- `19,193` vertices from `examples/data/bunny.node`
- `79,935` tetrahedra from `examples/data/bunny.ele`
- `20,832` extracted surface triangles
- `31,248` extracted surface edges

The mesh is normalized to a height of `0.35`. The default drop starts low and
visibly left/back at `(-0.28, 0.15, -0.20)`, advances for 500 steps with
`dt=0.01`, and retains `0.90` of the velocity at each step. The collision
floor is mathematically infinite. Its rendered plane is explicitly
`1000 x 1000`, while camera framing still depends only on the bunny
trajectory.

## Objective and design fields

All three optimization modes minimize the same final-frame position loss:

```text
L = 1/N sum(i=1..N) ||x[T,i] - (X_rest[i] + target_position)||².
```

There is no synthetic target trajectory and no target mass, Young's modulus,
or Poisson ratio. The default target is centered horizontally and places the
undeformed bunny's bottom at the contact activation height:

```text
target_position = (0, sqrt(d_hat), 0).
```

Use `--target-position X Y Z` to select a different target configuration.
The final-position objective is identical for every design. On a horizontal,
frictionless floor, material parameters cannot directly remove a uniform
rigid horizontal offset; initial-position optimization is therefore the mode
that produces the large horizontal correction.

The independently optimized controls are:

- `young`: one Young value and one Poisson value per tetrahedron; both fields
  are updated together.
- `mass`: one mass value per vertex. The default `4e-5` per vertex gives a
  total mass of `0.76772`.
- `initial-position`: one three-component translation shared by the initial
  vertex positions.

## Forward and backward systems

The example does not call `scene.addEnergy`, `scene.addMinimizeTarget`, or
`scene.minimizeEnergy`. Its forward Newton and backward adjoint operations use
the differentiated matrices directly:

```text
Phi[k+1](x; x[k], v[k], theta) =
    inertia(x, x[k] + retention h v[k] + h² gravity, vertex_mass)
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
frames                     = 500
velocity_retention         = 0.90
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

The measured large-mesh comparison uses the default 500 frames. This includes
the drop, first floor contact around frame 20, and the complete five-second
settling trajectory:

```bash
PYTHONPATH=yasps python examples/inverse_simulation/adjoint_bunny.py \
  --parameter all \
  --frames 500 \
  --optimization-steps 6 \
  --check-gradient \
  --video-directory examples/inverse_simulation/videos_large \
  --json-output examples/inverse_simulation/videos_large/results.json
```

The example uses NumPy, SciPy, PyCUDA, and PyVista; video encoding additionally
requires `ffmpeg`.

For each design, the renderer keeps all 501 side-by-side PNGs under
`videos_large/<design>/frames/` and writes a 501-frame, 1280x640 H.264 video
to `videos_large/<design>_before_after.mp4`. At 30 FPS each video lasts
16.7 seconds. It does not write OBJ intermediates.

Add `--check-gradient` and use one optimization step for a central
finite-difference comparison:

```bash
PYTHONPATH=yasps python examples/inverse_simulation/adjoint_bunny.py \
  --parameter all \
  --frames 500 \
  --optimization-steps 1 \
  --check-gradient
```

## Measured validation

The full 500-frame, six-step run produced:

| Design | Initial loss | Final loss | Reduction | Final design summary |
| --- | ---: | ---: | ---: | --- |
| Young + Poisson per element | 1.20168e-1 | 1.19288e-1 | 0.732% | means: Young `11086.58`, Poisson `0.322078` |
| Initial translation | 1.20168e-1 | 1.78452e-3 | 98.515% | `(-0.003378, 0.150025, -0.002333)` |
| Mass per vertex | 1.20168e-1 | 1.16435e-1 | 3.106% | mean `4.10979e-5`, total `0.788791` |

The optimized material, initial-position, and mass trajectories peaked at
`165`, `276`, and `491` self-contact pairs, and at `684`, `696`, and `673`
floor contacts, respectively. Their maximum stationarity residuals were
`1.94e-7`, `1.99e-7`, and `1.97e-7`, all below the `2e-7` contact
threshold. The initial 500-step forward trajectories took 11.5--12.1 seconds;
the final ones took 8.76--14.1 seconds.

Central directional finite-difference checks through all 500 frames measured
relative errors of:

- joint Young/Poisson: `3.86e-2`
- initial translation: `1.40e-5`
- per-vertex mass: `1.75e-1`

The material and mass checks are less tight because the perturbations switch
closest features and active contacts during the five-second trajectory.
Every accepted optimization update is nevertheless validated by a separate
500-step forward simulation and backtracked until its measured loss
decreases.
