#!/usr/bin/env bash
python setup.py build_ext --inplace --parallel $(nproc)
python setup.py bdist_wheel
pip install dist/*.whl --force-reinstall
# pip install .
