"""Run the unchanged separate-Jacobian partial-ABD setup on Metal."""

import os
from pathlib import Path
import runpy


directory = Path(__file__).resolve().parent
os.environ["YASPS_BACKEND"] = "metal"
os.chdir(directory)
(directory / "outputs").mkdir(exist_ok=True)
runpy.run_path(directory / "one_bunny_partial_abd.py", run_name="__main__")
