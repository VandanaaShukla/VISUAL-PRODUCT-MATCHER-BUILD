import os, glob, numpy as np, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.clip_utils import get_image_embedding
from core.faiss_manager import add_to_index, save_index  # sklearn version
from core import config

def iter_images(root):
    exts = (".jpg", ".jpeg", ".png")
    for path in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if path.lower().endswith(exts):
            yield path

def main():
    count = 0
    for img_path in iter_images(config.IMAGE_DIR):
        try:
            emb = get_image_embedding(img_path).astype(np.float32)  # (1,512)
            # (optional) normalize for cosine search if your faiss_manager uses cosine
            # emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
            add_to_index(emb, img_path)
            count += 1
        except Exception as e:
            print(f"Skip {img_path}: {e}")
    save_index()
    print(f"✅ Added {count} images from {config.IMAGE_DIR}")

if __name__ == "__main__":
    main()