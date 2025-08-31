# core/faiss_manager.py  — scikit-learn + cosine + scores
import os
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
from core import config

embedding_dim = 512
index_file = config.FIASS_INDEX_FILE   # keep your existing filenames
paths_file = config.IMAGE_PATH_FILE

_image_paths: list[str] = []
_embeddings: np.ndarray | None = None   # shape [N, 512] (normalized)
_nn: NearestNeighbors | None = None

def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / norm

def _save_state():
    os.makedirs(os.path.dirname(index_file), exist_ok=True)
    with open(index_file, "wb") as f:
        pickle.dump(_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(paths_file, "wb") as f:
        pickle.dump(_image_paths, f, protocol=pickle.HIGHEST_PROTOCOL)

def _rebuild_nn():
    global _nn
    if _embeddings is not None and len(_image_paths) == len(_embeddings) and len(_image_paths) > 0:
        _nn = NearestNeighbors(metric="cosine", n_neighbors=min(3, len(_image_paths)))
        _nn.fit(_embeddings)
    else:
        _nn = None

def _load_state():
    global _image_paths, _embeddings
    # paths
    if os.path.exists(paths_file):
        try:
            with open(paths_file, "rb") as f:
                _image_paths = pickle.load(f)
        except Exception:
            _image_paths = []
    else:
        _image_paths = []

    # embeddings (expect pickled numpy)
    _embeddings = None
    if os.path.exists(index_file):
        try:
            with open(index_file, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, np.ndarray):
                _embeddings = obj.astype(np.float32, copy=False)
            else:
                _embeddings = None
        except Exception:
            try: os.remove(index_file)
            except Exception: pass
            _embeddings = None

    _rebuild_nn()

# load on import and print count
_load_state()
print(f"Loaded {0 if _embeddings is None else len(_image_paths)} images from the index.")

def save_index():
    _save_state()

def add_to_index(embedding: np.ndarray, image_path: str):
    """Accepts (512,) or (1,512) CLIP vector; stores normalized."""
    global _embeddings, _image_paths
    emb = _normalize(embedding)
    if _embeddings is None:
        _embeddings = emb
    else:
        _embeddings = np.vstack([_embeddings, emb])
    _image_paths.append(image_path)
    _save_state()
    _rebuild_nn()

def search_similar(embedding: np.ndarray, k: int = 3):
    """
    Returns list of (image_path, similarity) where similarity ∈ [0,1].
    """
    if _nn is None or _embeddings is None or len(_image_paths) == 0:
        return []

    # L2-normalize query
    q = np.asarray(embedding, dtype=np.float32)
    if q.ndim == 1: q = q.reshape(1, -1)
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)

    k_eff = min(k, len(_image_paths))
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(metric="cosine", n_neighbors=k_eff)
    nn.fit(_embeddings)  # _embeddings must be normalized when added
    distances, indices = nn.kneighbors(q, n_neighbors=k_eff)

    out = []
    for d, i in zip(distances[0], indices[0]):
        sim = float(1.0 - d)  # cosine_similarity = 1 - cosine_distance
        out.append((_image_paths[int(i)], sim))
    return out


def reset_index():
    """Clears index files but keeps images (safer)."""
    global _embeddings, _image_paths, _nn
    _embeddings = None
    _image_paths = []
    _nn = None
    if os.path.exists(index_file): os.remove(index_file)
    if os.path.exists(paths_file): os.remove(paths_file)
    # If you really want to wipe images too, uncomment below:
    # import shutil
    # if os.path.exists(config.IMAGE_DIR):
    #     shutil.rmtree(config.IMAGE_DIR)
    #     os.makedirs(config.IMAGE_DIR, exist_ok=True)
def prune_missing_files():
    """Remove entries from the index whose image files no longer exist."""
    import os
    global _embeddings, _image_paths
    if _embeddings is None or not _image_paths:
        return 0
    keep = [i for i, p in enumerate(_image_paths) if os.path.exists(p)]
    if len(keep) == len(_image_paths):
        return len(_image_paths)
    _embeddings = _embeddings[keep] if _embeddings is not None else None
    _image_paths = [_image_paths[i] for i in keep]
    _save_state()
    _rebuild_nn()
    return len(_image_paths)
