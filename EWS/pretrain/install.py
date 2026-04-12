import os
import subprocess
import sys

# RTX 50 / Blackwell (sm_120, e.g. 5090): cu124/cu121 wheels often lack sm_120 and raise
# "no kernel image is available for execution on the device". Use CUDA 12.8+ builds (default: nightly cu128).
#
# Other GPUs may use stable cu128 without --pre, e.g.:
#   set TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
#   set TORCH_PIP_PRE=0
index_url = os.environ.get(
    "TORCH_INDEX_URL", "https://download.pytorch.org/whl/nightly/cu128"
)
extra = (
    ["--pre"]
    if os.environ.get("TORCH_PIP_PRE", "1") not in ("0", "false", "False")
    else []
)

subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--force-reinstall",
        *extra,
        "torch",
        "--index-url",
        index_url,
    ]
)
