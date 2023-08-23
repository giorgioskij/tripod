import os
from pathlib import Path
# import sys

ROOT: Path = Path(__file__).parent
os.chdir(ROOT)

CKP_PATH: Path = ROOT / "checkpoints"
DATA_PATH: Path = ROOT / "datasets"
# sys.path.append(str(ROOT))