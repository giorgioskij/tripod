import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

ROOT: Path = Path(__file__).parent
os.chdir(ROOT)

CKP_PATH: Path = ROOT / "checkpoints"
DATA_PATH: Path = ROOT / "datasets"
# sys.path.append(str(ROOT))

# main hyperparameters
