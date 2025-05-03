#!/usr/bin/env bash
python setup.py build_ext --inplace --parallel $(nproc)
python setup.py bdist_wheel
pip install dist/*.whl --force-reinstall
# pip install .

fcm_token=$(cat fcm_token)
gmtu --fcm-token $fcm_token --content "YASPS compile finished. GET THE FUCK BACK TO WORK!"
