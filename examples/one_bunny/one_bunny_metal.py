"""Run the unchanged one-bunny setup on the Metal backend."""

import os
from pathlib import Path
import runpy


directory = Path(__file__).resolve().parent
os.environ["YASPS_BACKEND"] = "metal"
os.chdir(directory)
(directory / "outputs").mkdir(exist_ok=True)
runpy.run_path(directory / "one_bunny.py", run_name="__main__")
