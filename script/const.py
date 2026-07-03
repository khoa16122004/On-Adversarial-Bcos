from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = str(Path(os.getenv("CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints")))
IMAGENET_VAL_DATA = os.getenv("IMAGENET_VAL_DATA", "/datastore/elo/quanphm/dataset/ImageNet1K/val")
ANNOTATIONS_FILE = os.getenv("ANNOTATIONS_FILE", str(PROJECT_ROOT / "script" / "id_2_classname.json"))
CLASSIFICATION_RESULT_DIR = os.getenv("CLASSIFICATION_RESULT_DIR", str(PROJECT_ROOT / "classification_result"))