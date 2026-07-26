"""Run the unchanged Brazil-nuts setup on the Metal backend."""

import os
from pathlib import Path
import runpy


directory = Path(__file__).resolve().parent
os.environ["YASPS_BACKEND"] = "metal"
os.chdir(directory)
for output in ("outputs", "meshes", "positions"):
  (directory / output).mkdir(exist_ok=True)
runpy.run_path(directory / "brazil_nuts.py", run_name="__main__")
