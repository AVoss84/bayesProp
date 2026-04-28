import os
from pathlib import Path
from typing import Optional

package_root = Path(__file__).resolve().parents[3]

defaults = {
    "CODE_DIR": str(package_root / "src"),
    "DATA_DIR": "",  # external
    "DATA_PKG_DIR": str(package_root / "data"),
}

# ------------------------------------------------------------

for env in defaults.keys():
    if env not in os.environ:
        os.environ[env] = defaults[env]

CODE_DIR = os.environ["CODE_DIR"]
DATA_DIR = os.environ["DATA_DIR"]
DATA_PKG_DIR = os.environ["DATA_PKG_DIR"]

# Setting for publishing the package artifacts to registry
# -----------------------------------------------------------
# package: Optional[str] = None
# if os.getenv("PUBLISH_ARTIFACTS", "False").lower() == "true":
#     package = "agbe_cc.config"
# -----------------------------------------------------------
