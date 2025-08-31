# download_ddg.py
import time, hashlib, mimetypes, re
from pathlib import Path
from io import BytesIO
import requests
from PIL import Image
from ddgs import DDGS

CATEGORIES = [
    "Nike Shoes","Adidas Shoes","Puma Shoes","Handbags","Watches",
    "Backpacks","Sunglasses","Jackets","T-shirts","Dresses",
    "Laptops","Smartphones","Headphones","Chairs","Tables",
    "Dogs","Cats","Birds","Cars","Bikes",
    "Plants","Flowers","Paintings","Sculptures","Cups"
]
PER_CLASS = 10
ROOT = Path("images"); ROOT.mkdir(exist_ok=True)
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

def safe_name(s: str) -> str:
    s = re.sub(r"[^\w\s-]+", "", s).strip()
    return re.sub(r"[\s-]+", " ", s)

def fetch_and_save(url: str, out_dir: Path, idx: int) -> bool:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=15)
        r.raise_for_status()
        ct = r.headers.get("Content-Type", "").split(";")[0].lower()
        ext = {"image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png", "image/webp": ".jpg"}.get(ct, "")
        if not ext:
            import mimetypes as mt
            guess = mt.guess_extension(url.split("?")[0])
            ext = guess if guess in (".jpg", ".jpeg", ".png") else ".jpg"
        h = hashlib.sha1(r.content).hexdigest()[:12]
        out_path = out_dir / f"Image_{idx:03d}_{h}{ext}"
        Image.open(BytesIO(r.content)).convert("RGB").save(out_path, quality=90)
        return True
    except Exception:
        return False

def ddg_images(query, need):
    # try multiple backends/regions with backoff
    tried = []
    with DDGS() as d:
        for backend in (None, "duckduckgo", "lite"):
            for region in ("us-en","wt-wt"):
                tried.append((backend, region))
                try:
                    it = d.images(query, max_results=need*4, region=region, safesearch="moderate", backend=backend)
                    results = list(it)
                    if results:
                        return results
                except Exception:
                    time.sleep(2)
    raise RuntimeError(f"No results for {query} (tried {tried})")

def download_category(q: str, k: int):
    q_dir = ROOT / safe_name(q)
    q_dir.mkdir(parents=True, exist_ok=True)
    have = len([p for p in q_dir.iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png")])
    target = max(k - have, 0)
    if target == 0:
        print(f"skip {q} (already have ≥{k})"); return

    print(f"↓ {q} (need {target})")
    got, attempt = 0, 0
    while got < target and attempt < 5:
        try:
            results = ddg_images(q, target-got)
            for item in results:
                url = item.get("image") or item.get("thumbnail")
                if not url:
                    continue
                if fetch_and_save(url, q_dir, have+got+1):
                    got += 1
                    if got >= target:
                        break
                time.sleep(0.3)
            if got < target:
                attempt += 1
                wait = min(10, 2**attempt)
                print(f"  partial ({got}/{target}). retry in {wait}s …")
                time.sleep(wait)
        except Exception as e:
            attempt += 1
            wait = min(15, 3**attempt)
            print(f"  error ({e}). retry in {wait}s …")
            time.sleep(wait)
    print(f"  +{got} files (now ≈{have+got})")
    time.sleep(2.0)

def main():
    for q in CATEGORIES:
        download_category(q, PER_CLASS)
    print("✅ Download complete.")

if __name__ == "__main__":
    main()
