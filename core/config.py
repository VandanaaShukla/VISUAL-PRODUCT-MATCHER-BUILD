# core/config.py
from pathlib import Path

# ---- Repo root (works on Windows & Linux) ----
REPO_ROOT = Path(__file__).resolve().parents[1]

# ---- Tunables ----
TOP_K = 3

# ---- Folders committed in the repo ----
IMAGE_DIR = REPO_ROOT / "images"
INDEX_DIR = REPO_ROOT / "manage_index"

# ---- Files ----
FIASS_INDEX_FILE = INDEX_DIR / "index_file.index"
IMAGE_PATH_FILE  = INDEX_DIR / "image_paths.pkl"

# ---- Ensure folders exist ----
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
