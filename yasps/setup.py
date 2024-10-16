from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

extensions = [
    Extension("yasps.scene", ["yasps/scene/scene.pyx"]),
    Extension("yasps.attribute", ["yasps/attribute/attribute.pyx"]),
    Extension("yasps.mesh", ["yasps/mesh/mesh.pyx"]),
    Extension("yasps.connectivity", ["yasps/connectivity/connectivity.pyx"]),
    Extension("yasps.primitive", ["yasps/primitive/primitive.pyx"]),
    Extension("yasps.operator", ["yasps/operator/operator.pyx"]),
    Extension("yasps.deviceKernel", ["yasps/kernel/deviceKernel.pyx"]),
    Extension("yasps.codeGenerator", ["yasps/codeGenerator/codeGenerator.pyx"]),
    Extension("yasps.globalKernel", ["yasps/kernel/globalKernel.pyx"]),
    Extension("yasps.energy", ["yasps/energy/energy.pyx"]),
    Extension("yasps.minimizer", ["yasps/minimizer/minimizer.pyx"]),
    Extension("yasps.autodiff", ["yasps/attribute/autodiff.pyx"]),
    Extension("yasps.helper", ["yasps/helper.pyx"]),

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
    ext_modules=cythonize(extensions),
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
