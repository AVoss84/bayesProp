import os
from pathlib import Path

# -------------------------------
# Which environment to use?
# -------------------------------
using = "vm"  # local machine
# using = "docker"  # container

package_root = Path(__file__).parent.parent.parent.parent

## Check if required environment variables exist
## if not apply default paths from test environment:
# -----------------------------------------------------------
if using == "vm":
    defaults = {
        "CODE_DIR": str(package_root / "src"),
        "DATA_DIR": "",  # external
        "DATA_PKG_DIR": str(
            package_root / "data"
        ),  # internal, i.e. data folder within package
        "APP_PORT": "5000",
        "APP_CONNECTION": "127.0.0.1",
    }
else:
    defaults = {
        "CODE_DIR": "/app/src/",
        "DATA_DIR": "/app/data/",
        "DATA_PKG_DIR": "/app/data/",  # data folder within package
        "APP_PORT": "8080",
        "APP_CONNECTION": "0.0.0.0",
    }
# -------------------------------------------------------------------------------------------------------------------------------

for env in defaults.keys():
    if env not in os.environ:
        os.environ[env] = defaults[env]

CODE_DIR = os.environ["CODE_DIR"]
DATA_DIR = os.environ["DATA_DIR"]
APP_PORT = os.environ["APP_PORT"]
APP_CONNECTION = os.environ["APP_CONNECTION"]
DATA_PKG_DIR = os.environ["DATA_PKG_DIR"]
