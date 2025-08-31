# core/config.py
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

TOP_K = 3
IMAGE_DIR = REPO_ROOT / "images"
INDEX_DIR = REPO_ROOT / "manage_index"

FIASS_INDEX_FILE = INDEX_DIR / "index_file.pkl"
  # pickled np.ndarray
IMAGE_PATH_FILE  = INDEX_DIR / "image_paths.pkl"    # list[str] (repo-relative)

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
