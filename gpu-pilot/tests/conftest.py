import os
import sys
from pathlib import Path

# Core Noesis is intentionally not installed or upgraded by this pilot.
root = Path(os.environ.get("NOESIS_SOURCE", Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(root))
