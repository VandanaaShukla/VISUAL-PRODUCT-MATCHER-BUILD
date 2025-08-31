# core/faiss_manager.py  — scikit-learn + cosine + scores
from __future__ import annotations
import pickle
import numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from core import config

embedding_dim = 512
index_file: Path = config.FIASS_INDEX_FILE
paths_file: Path = config.IMAGE_PATH_FILE

_image_paths: list[str] = []          # STORED as repo-relative POSIX strings
_embeddings: np.ndarray | None = None # shape [N, 512] (normalized)
_nn: NearestNeighbors | None = None

# ---------- path helpers ----------
def _to_abs(p: str | Path) -> Path:
    """Stored (relative or absolute) -> absolute path under repo root."""
    s = str(p).replace("\\", "/")
    q = Path(s)
    return q if q.is_absolute() else (config.REPO_ROOT / q)

def _to_rel(p: str | Path) -> str:
    """Any path -> repo-relative POSIX string (for storing in pickle)."""
    q = Path(p)
    try:
        return str(q.resolve().relative_to(config.REPO_ROOT)).replace("\\", "/")
    except Exception:
        return q.name.replace("\\", "/")

def _to_ui(p: str | Path) -> str:
    """Path -> absolute POSIX string for Streamlit image()"""
    return _to_abs(p).as_posix()

# ---------- math helpers ----------
def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / norm

# ---------- persistence ----------
def _save_state():
    index_file.parent.mkdir(parents=True, exist_ok=True)
    with open(index_file, "wb") as f:
        pickle.dump(_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(paths_file, "wb") as f:
        # always store relative POSIX strings
        pickle.dump([_to_rel(p) for p in _image_paths], f, protocol=pickle.HIGHEST_PROTOCOL)

def _rebuild_nn():
    global _nn
    if _embeddings is not None and len(_image_paths) > 0 and len(_image_paths) == len(_embeddings):
        _nn = NearestNeighbors(metric="cosine", n_neighbors=min(3, len(_image_paths)))
        _nn.fit(_embeddings)
    else:
        _nn = None

def _load_state():
    global _image_paths, _embeddings
    # paths list
    if paths_file.exists():
        try:
            with open(paths_file, "rb") as f:
                raw = pickle.load(f)
            # normalize to relative POSIX in memory
            _image_paths = [_to_rel(p) for p in (raw or [])]
        except Exception:
            _image_paths = []
    else:
        _image_paths = []

    # embeddings
    _embeddings = None
    if index_file.exists():
        try:
            with open(index_file, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, np.ndarray):
                _embeddings = obj.astype(np.float32, copy=False)
        except Exception:
            try:
                index_file.unlink()
            except Exception:
                pass
            _embeddings = None

    _rebuild_nn()

# load on import and print count
_load_state()
print(f"Loaded {0 if _embeddings is None else len(_image_paths)} images from the index.")

# ---------- public API ----------
def save_index():
    _save_state()

def add_to_index(embedding: np.ndarray, image_path: str | Path):
    """Accepts (512,) or (1,512) CLIP vector; stores normalized + path (relative)."""
    global _embeddings, _image_paths
    emb = _normalize(embedding)
    _embeddings = emb if _embeddings is None else np.vstack([_embeddings, emb])
    _image_paths.append(_to_rel(image_path))
    _save_state()
    _rebuild_nn()

def search_similar(embedding: np.ndarray, k: int = 3):
    """
    Returns list of (ui_abs_path, similarity) where similarity ∈ [0,1].
    """
    if _nn is None or _embeddings is None or not _image_paths:
        return []

    q = np.asarray(embedding, dtype=np.float32)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)

    k_eff = min(k, len(_image_paths))
    nn = NearestNeighbors(metric="cosine", n_neighbors=k_eff)
    nn.fit(_embeddings)
    distances, indices = nn.kneighbors(q, n_neighbors=k_eff)

    out = []
    for d, i in zip(distances[0], indices[0]):
        sim = float(1.0 - d)
        p = _image_paths[int(i)]
        if _to_abs(p).exists():
            out.append((_to_ui(p), sim))
    return out

def reset_index():
    """Clears index files but keeps images (safer)."""
    global _embeddings, _image_paths, _nn
    _embeddings = None
    _image_paths = []
    _nn = None
    if index_file.exists(): index_file.unlink()
    if paths_file.exists(): paths_file.unlink()

def prune_missing_files():
    """Remove entries whose image files no longer exist."""
    global _embeddings, _image_paths
    if _embeddings is None or not _image_paths:
        return 0
    keep = [i for i, p in enumerate(_image_paths) if _to_abs(p).exists()]
    if len(keep) == len(_image_paths):
        return len(_image_paths)
    _embeddings = _embeddings[keep] if _embeddings is not None else None
    _image_paths = [_image_paths[i] for i in keep]
    _save_state()
    _rebuild_nn()
    return len(_image_paths)
