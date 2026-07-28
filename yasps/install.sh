#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

python_bin="${PYTHON:-python}"

"${python_bin}" -m pip install setuptools wheel Cython numpy pycuda

"${python_bin}" setup.py build_ext --inplace --parallel "$(nproc)"
"${python_bin}" setup.py bdist_wheel

wheel_path=""
for candidate in dist/yasps-*.whl; do
  [[ -e "${candidate}" ]] || continue
  if [[ -z "${wheel_path}" || "${candidate}" -nt "${wheel_path}" ]]; then
    wheel_path="${candidate}"
  fi
done
if [[ -z "${wheel_path}" ]]; then
  echo "No YASPS wheel was produced in ${script_dir}/dist." >&2
  exit 1
fi

"${python_bin}" -m pip install --force-reinstall --no-deps "${wheel_path}"

if [[ -f fcm_token ]] && command -v gmtu >/dev/null 2>&1; then
  gmtu \
    --fcm-token "$(cat fcm_token)" \
    --content "YASPS compile finished. GET THE FUCK BACK TO WORK!"
fi
