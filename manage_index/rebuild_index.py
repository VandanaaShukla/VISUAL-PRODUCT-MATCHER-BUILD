# manage_index/rebuild_index.py — rebuild portable index for both local & cloud
from pathlib import Path
import pickle
import numpy as np
from core import config
from core.clip_utils import get_image_embedding

def iter_images(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            # store repo-relative POSIX path
            yield str(p.resolve().relative_to(config.REPO_ROOT)).replace("\\", "/")

def main():
    img_dir = config.IMAGE_DIR
    if not img_dir.exists():
        print("Images folder not found:", img_dir)
        return

    rel_paths = list(iter_images(img_dir))
    if not rel_paths:
        print("No images found under:", img_dir)
        return

    embs = []
    kept = []
    for rel in rel_paths:
        abs_path = (config.REPO_ROOT / rel).as_posix()
        try:
            emb = get_image_embedding(abs_path)          # (512,) or (1,512)
            v = np.asarray(emb, dtype=np.float32).reshape(1, -1)
            # L2-normalize so cosine kNN works well
            v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
            embs.append(v)
            kept.append(rel)
        except Exception as e:
            print("SKIP:", rel, "|", e)

    if not embs:
        print("No embeddings computed.")
        return

    E = np.vstack(embs).astype(np.float32, copy=False)
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    with open(config.FIASS_INDEX_FILE, "wb") as f:
        pickle.dump(E, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(config.IMAGE_PATH_FILE, "wb") as f:
        pickle.dump(kept, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved embeddings: {E.shape} -> {config.FIASS_INDEX_FILE}")
    print(f"Saved paths: {len(kept)} -> {config.IMAGE_PATH_FILE}")

if __name__ == "__main__":
    main()
