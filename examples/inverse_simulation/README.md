# Discrete-adjoint bunny drop

`adjoint_bunny.py` is a complete implicit inverse-simulation example on the
repository's full TetGen bunny:

- `19,193` vertices from `examples/data/bunny.node`
- `79,935` tetrahedra from `examples/data/bunny.ele`
- `20,832` extracted surface triangles
- `31,248` extracted surface edges

The mesh is normalized to a height of `0.35`. The default drop starts low and
visibly left/back at `(-1.28, 0.15, -0.20)` and advances for 500 steps with
`dt=0.01`. There is no artificial velocity damping. The collision floor is
mathematically infinite. Its rendered plane is explicitly `1000 x 1000`,
while camera framing still depends only on the bunny trajectory.

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

The mass and material controls are shared by the complete trajectory, not
only its first frame. The differentiation graphs for `C` are constructed
once, but their numerical values are recomputed at every restored timestep:

```text
C[k] = d² Phi[k] / d(position[k+1]) d(design)
design_gradient = -sum(k=0..T-1) C[k]ᵀ mu[k].
```

Mass uses the inertia energy's `C[k]`; Young and Poisson use the elastic
energy's `C[k]`. Initial-position optimization is different because its
control enters the initial state directly, so its gradient is the accumulated
initial-state adjoint rather than a per-frame `C[k]` product. The implicit
adjoint differentiates each converged frame once; it does not backpropagate
through the internal Newton iterations.

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
d_hat                      = 1e-6
max_newton_iterations      = 300
correction_rate_tolerance  = 1e-2
newton_tolerance           = 1e-12
contact_newton_tolerance   = 2e-7
```

As in `dropping_in_container`, a frame's Newton correction must satisfy
`max(abs(correction)) / dt < 1e-2`. The example additionally requires the
stationarity residual to meet the smooth/contact threshold, because the
implicit adjoint differentiates the converged stationarity equation rather
than an unfinished Newton iterate. The looser contact threshold handles
closest-feature switching at impact. Newton is hard-capped at 300 iterations.
Only at that hard cap, a contact frame may terminate in the narrow residual
stagnation band up to `3e-7` if its correction rate already meets `1e-2`;
the ordinary `2e-7` threshold is unchanged during the preceding iterations.

## Outer optimization

There is no design-level acceptance test or backtracking line search. Each
outer round performs exactly:

1. one forward simulation whose converged states, velocities, and contacts
   are checkpointed;
2. one reverse adjoint sweep through all checkpoints;
3. one normalized and bounded design update; and
4. one new forward simulation from that updated design.

The new design is retained even when the measured loss increases. Optimization
stops after 20 updates or when the absolute loss reaches `1e-4`. The per-frame
Newton solve and its energy line search remain in place; they are part of the
implicit forward integrator, not an outer design line search.

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
  --optimization-steps 20 \
  --video-directory examples/inverse_simulation/videos_large \
  --json-output examples/inverse_simulation/videos_large/results.json
```

The example uses NumPy, SciPy, PyCUDA, and PyVista; video encoding additionally
requires `ffmpeg`.

For each design, the renderer keeps all 501 side-by-side PNGs under
`videos_large/<design>/frames/` and writes a 501-frame, 1280x640 H.264 video
to `videos_large/<design>_before_after.mp4`. By default, the playback rate is
`round(1 / dt) = 100` FPS, so the video lasts 5.01 seconds and follows
simulation time. Each bunny is translucent and contains an animated cyan
vertex-centroid marker. The renderer does not write OBJ intermediates.

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

The full 500-frame, 20-update fixed-step comparison produced:

| Design | Initial loss | Round-20 loss | Reduction | Best observed round/loss | Increasing updates |
| --- | ---: | ---: | ---: | ---: | ---: |
| Young + Poisson per element | 1.715823 | 1.707982 | 0.457% | 19 / 1.706975 | 9 |
| Initial translation | 1.715821 | 0.0500833 | 97.081% | 11 / 0.0397940 | 8 |
| Mass per vertex | 1.715824 | 1.596730 | 6.941% | 18 / 1.565240 | 8 |

The round-20 values are the actual retained designs; the best intermediate
values are reported only for comparison. The final initial translation is
`(-0.015558, 0.154651, -0.106151)`. The final material means are Young
`10586.759` and Poisson `0.318490`. The final mean vertex mass is
`6.26370e-5`, for total mass `1.20219`.

Final-trajectory convergence and contact statistics were:

| Design | Max residual | Max correction rate | Mean Newton iterations/frame | Max self pairs | Max floor contacts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Young + Poisson | 1.956e-7 | 9.741e-3 | 2.958 | 1,189 | 889 |
| Initial translation | 1.993e-7 | 9.196e-3 | 2.582 | 983 | 916 |
| Mass | 2.000e-7 | 9.970e-3 | 4.834 | 2,743 | 854 |

All correction rates remain below `1e-2`; all reported final residuals remain
below the ordinary `2e-7` contact threshold.

### Real simulation timings

Each mode performs 21 forward trajectories and 20 backward sweeps:

| Design | Total forward | Total backward | Total optimization | Mean forward | Mean backward |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial translation | 762.94 s | 848.08 s | 1,611.06 s | 36.33 s | 42.40 s |
| Young + Poisson | 726.61 s | 1,515.24 s | 2,242.49 s | 34.60 s | 75.76 s |
| Mass | 1,247.20 s | 2,370.83 s | 3,618.38 s | 59.39 s | 118.54 s |

The matrix logs include the complete forward and exact-adjoint Hessian
assemblies:

| Design | Hessian assemblies | Total Hessian time | Mean | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Initial translation | 46,302 | 321.681 s | 6.947 ms | 165.24 ms |
| Young + Poisson | 44,467 | 306.206 s | 6.886 ms | 167.04 ms |
| Mass | 63,144 | 459.492 s | 7.277 ms | 163.88 ms |

For every one of the `500 x 20 = 10,000` reverse timesteps, the implementation
also recomputes both state-transition mixed matrices. This gives 20,000
mixed-matrix computations for initial-position optimization. Material and
mass each add one `C[k]` computation per reverse timestep, giving 30,000:

| Design | Mixed computations | Total time | Mean | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Initial translation (`B_x`, `B_v`) | 20,000 | 4.911 s | 0.246 ms | 14.75 ms |
| Young + Poisson (`B_x`, `B_v`, `C`) | 30,000 | 120.594 s | 4.020 ms | 41.49 ms |
| Mass (`B_x`, `B_v`, `C`) | 30,000 | 6.739 s | 0.225 ms | 10.61 ms |

Focused central directional finite-difference checks through all 500 frames
measured relative errors of `0.717%` for initial translation, `5.26%` for
mass, and `44.2%` for joint Young/Poisson. The contact problem is only
piecewise smooth: perturbing distributed material fields changes closest
features and active contact pairs, which makes the material finite-difference
comparison particularly sensitive. Production optimization did not enable
`--check-gradient`, so it incurred no finite-difference trajectories.
