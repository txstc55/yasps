# Discrete-adjoint bunny drop

`adjoint_bunny.py` is a complete implicit inverse-simulation example. It
loads and tetrahedralizes the repository bunny, drops it for 200 steps of
0.01 seconds, records every converged state and active contact set, and then
walks the trajectory backward with the discrete adjoint method.

The example intentionally does not call `scene.addEnergy`,
`scene.addMinimizeTarget`, or `scene.minimizeEnergy`. Its forward Newton
steps and backward adjoint steps use the differentiated objects directly:

```text
A = d² Phi / d(position)²
B = d² Phi / d(position) d(previous_position)
C = d² Phi / d(position) d(parameter)

Aᵀ mu = lambda_next
lambda_previous = -Bᵀ mu
parameter_gradient += -Cᵀ mu
```

Every mixed second-order term is assembled in its retained chain-rule form,

```text
J_rowᵀ H_inner J_column + H_recursive,
```

so the two outer Jacobians and inner Hessian remain explicit. Constants that
are not differentiation targets contribute zero, even when they occur inside
an expression.

The forward Newton globalization uses a PSD-projected Neo-Hookean Hessian.
Self-contact barriers use closest-feature weights and normals lagged from
collision detection, making separation linear in position and their Newton
Hessians PSD. At convergence, the backward pass rebuilds the exact Hessian of
the stationarity residual, including exact nonlinear elastic and contact
terms. This follows the converged-stationarity prescription in
`Adjoint_Method.pdf`, rather than differentiating the intermediate Newton
iterations.

The position-only step is first order in time so the state used by the
adjoint is complete; there is no untracked velocity state. The terminal loss
is the mean squared norm of the final vertex positions relative to a centered
resting bunny.

Both floor barrier contacts and bunny self-contact use dynamic primitives.
The forward pass stores their connectivity and lagged closest-feature
linearization at every converged frame, and the backward pass restores the
contact state before rebuilding the exact Hessian and mixed Jacobians.

Run all three optimization examples:

```bash
PYTHONPATH=yasps python examples/inverse_simulation/adjoint_bunny.py
```

Or run one design variable:

```bash
PYTHONPATH=yasps python examples/inverse_simulation/optimize_bunny_young.py
PYTHONPATH=yasps python examples/inverse_simulation/optimize_bunny_initial_position.py
PYTHONPATH=yasps python examples/inverse_simulation/optimize_bunny_mass.py
```

Useful validation options:

```bash
# Short smoke run, including a central-difference adjoint check
PYTHONPATH=yasps python examples/inverse_simulation/adjoint_bunny.py \
  --parameter mass --frames 5 --optimization-steps 1 --check-gradient

# Full 200-frame result as JSON
PYTHONPATH=yasps python examples/inverse_simulation/adjoint_bunny.py \
  --frames 200 --dt 0.01 --json-output /tmp/adjoint_bunny_results.json
```

`pyvista` and `tetgen` are used only to make a small, deterministic
tetrahedral bunny from `examples/data/bunny_small.obj`. The default reduction
keeps the full forward/backward example practical while preserving the bunny
surface for collision detection.

## Validation

A 200-frame, `dt=0.01` run on the default 66-vertex validation mesh reached a
maximum stationarity residual below `1.0e-8`, exercised up to 12 self-contact
pairs and 5 floor contacts, and completed all Hessian solves without a
non-SPD failure. With two accepted adjoint updates, the observed results were:

| Design | Loss before | Loss after | Center distance before | Center distance after |
| --- | ---: | ---: | ---: | ---: |
| Young's modulus | 0.0290690 | 0.0288038 | 0.169359 | 0.168765 |
| Initial position | 0.0290690 | 0.00100921 | 0.169359 | 0.0252365 |
| Mass | 0.0290690 | 0.0287970 | 0.169359 | 0.168745 |

The initial-position example is the direct centering control: after only two
adjoint solves, its horizontal translation is approximately
`(0.00732, 0.00230)`, its horizontal center distance is `0.00767`, and its
center-distance loss drops by about 96.5%.
