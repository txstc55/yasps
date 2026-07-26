"""Run the unchanged dropping-in-container setup on the Metal backend."""

import os
from pathlib import Path
import runpy


directory = Path(__file__).resolve().parent
os.environ["YASPS_BACKEND"] = "metal"
os.chdir(directory)
for output in ("outputs", "meshes"):
  (directory / output).mkdir(exist_ok=True)
runpy.run_path(directory / "dropping_in_container.py", run_name="__main__")
