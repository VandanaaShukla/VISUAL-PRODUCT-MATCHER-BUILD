# core/index_manager.py
import pickle
import numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from core import config

embedding_dim = 512
index_file = config.FIASS_INDEX_FILE
paths_file = config.IMAGE_PATH_FILE

_image_paths: list[str] = []
_embeddings: np.ndarray | None = None
_nn: NearestNeighbors | None = None

def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

def _load_state():
    global _image_paths, _embeddings
    # Load paths
    if paths_file.exists():
        with open(paths_file, "rb") as f:
            _image_paths = pickle.load(f)
    else:
        _image_paths = []

    # Load embeddings
    if index_file.exists():
        with open(index_file, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, np.ndarray):
            _embeddings = _normalize(obj.astype(np.float32, copy=False))
        else:
            _embeddings = None
    else:
        _embeddings = None

_load_state()

def search_similar(embedding: np.ndarray, k: int = 3):
    if _embeddings is None or not _image_paths:
        return []

    q = _normalize(embedding)
    k_eff = min(k, len(_image_paths))

    nn = NearestNeighbors(metric="cosine", n_neighbors=k_eff)
    nn.fit(_embeddings)
    distances, indices = nn.kneighbors(q, n_neighbors=k_eff)

    out = []
    for d, i in zip(distances[0], indices[0]):
        sim = float(1.0 - d)  # cosine similarity
        out.append((_image_paths[int(i)], sim))
    return out
