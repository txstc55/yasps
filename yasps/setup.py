from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

extensions = [
    Extension("yasps.scene", ["yasps/scene/scene.pyx"]),
    Extension("yasps.attribute", ["yasps/attribute/attribute.pyx"]),
    Extension("yasps.mesh", ["yasps/mesh/mesh.pyx"]),
    Extension("yasps.connectivity", ["yasps/connectivity/connectivity.pyx"]),
    Extension("yasps.primitive", ["yasps/primitive/primitive.pyx"]),
    Extension("yasps.primitiveUnion", ["yasps/primitive/primitiveUnion.pyx"]),
    Extension("yasps.operator", ["yasps/operator/operator.pyx"]),
    Extension("yasps.deviceKernel", ["yasps/kernel/Compute/deviceKernel.pyx"]),
    Extension("yasps.codeGenerator", ["yasps/codeGenerator/codeGenerator.pyx"]),
    Extension("yasps.globalKernel", ["yasps/kernel/Compute/globalKernel.pyx"]),
    Extension("yasps.vector", ["yasps/matrixAndVector/vector/vector.pyx"]),
    Extension("yasps.matrix", ["yasps/matrixAndVector/matrix/matrix.pyx"]),
    Extension("yasps.gradientIndicesKernel", ["yasps/kernel/Coordinate/gradientIndicesKernel.pyx"]),
    Extension("yasps.gradient", ["yasps/matrixAndVector/vector/gradient.pyx"]),
    Extension("yasps.hessianAndGradientKernel", ["yasps/kernel/Hessian/hessianAndGradientKernel.pyx"]),
    Extension("yasps.hessian", ["yasps/matrixAndVector/matrix/hessian.pyx"]),
    Extension("yasps.secondOrderJacobian", ["yasps/matrixAndVector/matrix/secondOrderJacobian.pyx"]),
    Extension("yasps.coordinateCompressionKernel", ["yasps/kernel/Coordinate/coordinateCompressionKernel.pyx"]),
    Extension("yasps.diagonalBlockInverseKernel", ["yasps/kernel/Solver/diagonalBlockInverseKernel.pyx"]),
    Extension("yasps.solverKernel", ["yasps/kernel/Solver/solverKernel.pyx"]),
    Extension("yasps.solver", ["yasps/solver/solver.pyx"]),
    Extension("yasps.energy", ["yasps/energy/energy.pyx"]),
    Extension("yasps.minimizer", ["yasps/minimizer/minimizer.pyx"]),
    Extension("yasps.autodiff", ["yasps/attribute/autodiff.pyx"]),
    Extension("yasps.helper", ["yasps/helper.pyx"]),
    Extension("yasps.attributeHelper", ["yasps/attribute/attributeHelper.pyx"]),
    Extension("yasps.attributeOperations", ["yasps/attribute/attributeOperations.pyx"]),
    Extension("yasps.path", ["yasps/differentiator/path.pyx"]),
    Extension("yasps.differentiator", ["yasps/differentiator/differentiator.pyx"]),
    Extension("yasps.context", ["yasps/context/context.pyx"]),
    Extension("yasps.hessianKernelHeader", ["yasps/kernel/Hessian/hessianKernelHeader.pyx"]),
    Extension("yasps.hessianKernelFullProject", ["yasps/kernel/Hessian/hessianKernelFullProject.pyx"]),
    Extension("yasps.hessianKernelNoProject", ["yasps/kernel/Hessian/hessianKernelNoProject.pyx"]),
    Extension("yasps.hessianKernelHost", ["yasps/kernel/Hessian/hessianKernelHost.pyx"]),
    Extension("yasps.placementReorderKernel", ["yasps/kernel/Coordinate/placementReorderKernel.pyx"]),
    Extension("yasps.hessianKernelSeparateJacobian", ["yasps/kernel/Hessian/hessianKernelSeparateJacobian.pyx"]),
]
setup(
    name='yasps',  # This is the name of your package
    version='0.1.0',  # The initial release version
    author='NOT YET',  # Your name
    author_email='your.email@example.com',  # Your email
    description='A short description of the package',  # A short description
    # long_description=open('README.md').read(),  # A long description from your README file
    long_description_content_type='text/markdown',  # The content type of the long description
    url='https://github.com/yourusername/yasps',  # The URL to the repository
    packages=find_packages(),  # Finds all packages in the directory
    package_data={'yasps': ['*.txt', '*.cuh', '*.cu']},
    ext_modules=cythonize(
      extensions,
      annotate=False,        # Generates the HTML .html annotation files
      compiler_directives={"language_level": "3"},
      nthreads=16,       # optional, parallel compile
      force=False       # important! only rebuild changed files
    ),
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.org/classifiers/
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'numpy',
        'pycuda'
    ],
    python_requires='>=3.6',  # Minimum version requirement of the package
    include_package_data=True,  # Includes files described by MANIFEST.in
)
