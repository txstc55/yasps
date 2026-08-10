# Ideas

In this doc I will write down some ideas directly related to how YASPS can be improved

## Incorporating Sympy
Sympy, when doing differentiation and simplification, is quite slow. However its simplified computation tree is very nice. It would be nice to utilize Sympy's simplification feature to YASPS.

## Scene Duplication
How can we execute multiple scene, multiple minimizations in parallel, while reusing all the kernels, the computation trees, the differentiations? It's mostly engineering but how the frontend should be is also worth considering. This will also be useful for inverse simulation, where we actually want to run multiple simulations in parallel, and pick the best one, like a genetic algorithm.

## MAS Preconditioner
A good preconditioner is essentially trying to provide as much information about the inverse of the matrix itself as possible. MAS is a good candidate for this, and technically we can use it directly on the matrix structure without knowing what "mesh" is. This is different from AMG or just using the mesh structure to build a preconditioner.

## Permutation
This could work with MAS preconditioner. The idea is to permute the matrix so that the non-zero entries are more clustered together, which can improve the performance of iterative solvers. This could be done by analyzing the sparsity pattern of the matrix and finding an optimal permutation through METIS. This can also theoratically increase the performance of SpMV and other operations on the matrix.

## Dynamic Attributes
Right now we allow connectivity to be dynamic at runtime. However we do not allow the base attributes, the total number of it to be changed. So if something like remashing happened, which changes the matrix size itself, YASPS will not be able to handle it. We should allow the base attributes to be dynamic as well. All the backends are there, it's just we need to allow user to directly signal the change of the base attributes.

## Incorporate CCD
This is a small thing, make CCD an actual feature of YASPS instead of putting it in the examples folder.
