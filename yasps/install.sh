#!/usr/bin/env bash
python setup.py build_ext --inplace --parallel $(nproc)
pip install .
