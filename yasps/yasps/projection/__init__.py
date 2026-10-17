"""Small-EVD projection source and generated CUDA cache version."""

from hashlib import sha256
from pathlib import Path

SMALL_EVD_HEADER = Path(__file__).with_name("small_evd.cuh").read_text(encoding="utf-8")
# Direct row-major lower input; PSD prechecks remain bypassed.
PROJECTION_CACHE_TAG = "evd_reuse_direct_lower_v2_" + sha256(
    SMALL_EVD_HEADER.encode("utf-8")
).hexdigest()[:12]
