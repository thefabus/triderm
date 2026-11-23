"""Vision branch: frozen backbones, wound dataset, attention pooling head and its training."""

import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")  # no update check when importing
