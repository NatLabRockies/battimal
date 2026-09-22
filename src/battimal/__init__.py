import os
from pathlib import Path

__all__ = ["data", "models"]

from . import data
from . import models

BATTIMAL_DIR = Path(os.path.dirname(os.path.abspath(__file__)))