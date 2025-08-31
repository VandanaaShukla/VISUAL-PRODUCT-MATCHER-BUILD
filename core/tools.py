# tools/fix_paths.py
import pickle
from pathlib import Path
import config

P = config.IMAGE_PATH_FILE
root = config.REPO_ROOT

with open(P, "rb") as f:
    paths = pickle.load(f)

def to_rel(p: str) -> str:
    s = str(p).replace("\\", "/")
    q = Path(s)
    try:
        return str(q.resolve().relative_to(root)).replace("\\", "/")
    except Exception:
        return s if not q.is_absolute() else q.name.replace("\\", "/")

new_paths = [to_rel(p) for p in paths]

missing = [p for p in new_paths if not (root / p).exists()]
print("Total:", len(new_paths), "| Missing after fix:", len(missing))
if missing[:10]: print("Examples:", missing[:10])

with open(P, "wb") as f:
    pickle.dump(new_paths, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Wrote fixed:", P)
