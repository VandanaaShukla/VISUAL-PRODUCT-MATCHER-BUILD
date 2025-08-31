# streamlit_ui/app.py
import os
import sys
import logging
import requests
from io import BytesIO
from PIL import Image
import streamlit as st

# Quiet Streamlit watcher noise
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

# Allow "from core import ..." imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.clip_utils import get_image_embedding
from core.index_manager import search_similar, prune_missing_files  # optional auto-prune
from core.storage import save_uploaded_image, remove_uploaded_image
from core import config

# ---------------- Utilities ----------------
TMP_DIR = os.path.join(config.INDEX_DIR, "tmp_uploads")
os.makedirs(TMP_DIR, exist_ok=True)

# Version-compat image helper: new API on cloud, old API locally
def show_image(img_or_path, caption=None):
    try:
        # Newer Streamlit (≥1.49) accepts string widths
        st.image(img_or_path, caption=caption, width="stretch")
    except TypeError:
        # Older Streamlit (≤1.48) needs the legacy flag
        st.image(img_or_path, caption=caption, use_container_width=True)

def save_image_from_url(url: str) -> str:
    """Download an image from URL, save in temp, return local path."""
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    # choose extension by format if available, else jpg
    ext = (img.format or "JPEG").lower()
    if ext not in ("jpeg", "jpg", "png", "webp"):
        ext = "jpg"
    fname = os.path.join(TMP_DIR, f"url_{abs(hash(url))}.{ext}")
    img.save(fname, "JPEG" if ext in ("jpg", "jpeg") else ext.upper())
    return fname

def _resolve_ui_path(p: str) -> str:
    """Ensure a path is absolute (rooted at repo) and POSIX-like for Streamlit."""
    from pathlib import Path
    q = Path(p)
    if not q.is_absolute():
        q = Path(config.REPO_ROOT) / q
    return q.as_posix()

# ---------------- UI SETUP ----------------
st.set_page_config(page_title="Visual Product Matcher", layout="wide")

# Optional: prune dead paths silently on load so results never crash
try:
    _ = prune_missing_files()
except Exception:
    pass

# Custom styling
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
      html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
      .hero {
        background: linear-gradient(135deg, rgba(37,99,235,0.08), rgba(16,185,129,0.06));
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 18px;
      }
      .hero h1 { margin: 0; font-size: 28px; line-height: 1.3; color: #111827; }
      .hero p { margin: 6px 0 0 0; color: #374151; font-size: 15px; }
      .box {
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 18px;
        background: #ffffff;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
      }
      .cta-btn button {
        width: 100%;
        border-radius: 10px !important;
        font-weight: 600 !important;
        height: 44px !important;
        border: none !important;
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #fff !important;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.3) !important;
      }
      .cta-btn button:hover { filter: brightness(1.07); }
      .caption { color:#4b5563; font-size:0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
    <div class="hero">
      <h1>Visual Product Matcher Build</h1>
      <p>Upload an image <b>or paste an image URL</b> and instantly see the most similar items from the dataset.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Keep state for URL image so it persists between button clicks
if "url_image_path" not in st.session_state:
    st.session_state.url_image_path = None
if "url_embedding" not in st.session_state:
    st.session_state.url_embedding = None

# Layout: left (upload/preview) | right (controls/results)
left, right = st.columns([1, 2], gap="large")

# ---------------- LEFT: Upload OR URL Preview & Embedding ----------------
with left:
    st.markdown("<div class='box'>", unsafe_allow_html=True)
    st.subheader("1) Provide an image")

    tabs = st.tabs(["Upload", "From URL"])

    # ---- Upload Tab ----
    with tabs[0]:
        uploaded_file = st.file_uploader("Choose a JPG/PNG", type=["jpg", "jpeg", "png"], key="query_image")
        uploaded_preview_path = None
        uploaded_embedding = None

        if uploaded_file is not None:
            uploaded_preview_path = save_uploaded_image(uploaded_file)
            show_image(uploaded_preview_path, caption="Uploaded image")
            with st.spinner("Extracting features…"):
                try:
                    uploaded_embedding = get_image_embedding(uploaded_preview_path)
                    st.success("Upload ready for search ✓")
                except Exception as e:
                    st.error(f"Failed to compute embedding: {e}")

    # ---- URL Tab ----
    with tabs[1]:
        url = st.text_input("Paste an image URL (http/https)")
        colu1, colu2 = st.columns([1, 1])
        load_clicked = colu1.button("Load URL Image")
        clear_clicked = colu2.button("Clear URL")

        if clear_clicked:
            # clear previous URL state & temp file
            if st.session_state.url_image_path and os.path.exists(st.session_state.url_image_path):
                try:
                    os.remove(st.session_state.url_image_path)
                except Exception:
                    pass
            st.session_state.url_image_path = None
            st.session_state.url_embedding = None

        if load_clicked and url.strip():
            with st.spinner("Fetching image…"):
                try:
                    path = save_image_from_url(url.strip())
                    st.session_state.url_image_path = path
                    # show preview
                    show_image(path, caption="URL image")
                    # embed
                    with st.spinner("Extracting features…"):
                        st.session_state.url_embedding = get_image_embedding(path)
                    st.success("URL image ready for search ✓")
                except Exception as e:
                    st.error(f"Failed to load from URL: {e}")

        # If already loaded earlier, show preview again
        if st.session_state.url_image_path and os.path.exists(st.session_state.url_image_path):
            show_image(st.session_state.url_image_path, caption="URL image")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- RIGHT: Controls & Results ----------------
with right:
    st.markdown("<div class='box'>", unsafe_allow_html=True)
    st.subheader("2) Find similar images")


# --- Deployed commit & folder listing (debug) ---
from pathlib import Path
import subprocess, shlex

def _git(cmd: str):
    try:
        return subprocess.check_output(shlex.split(cmd)).decode().strip()
    except Exception as e:
        return f"ERR: {e}"

st.write("Deployed commit:", _git("git rev-parse HEAD"))
st.write("Index path:", Path(config.FIASS_INDEX_FILE).as_posix())
st.write("manage_index contents:", [p.name for p in Path(config.INDEX_DIR).glob("*")])
try:
    st.write("Index size (bytes):", Path(config.FIASS_INDEX_FILE).stat().st_size)
except Exception:
    st.write("Index size (bytes): <not found>")

    # ---- DEBUG DIAGNOSTICS (temporary; remove once OK) ----
    from pathlib import Path
    import pickle as _p
    

    st.caption("Diagnostics (temporary)")
    st.write("Repo root:", config.REPO_ROOT.as_posix() if hasattr(config.REPO_ROOT, "as_posix") else str(config.REPO_ROOT))
    st.write("IMAGE_DIR exists:", Path(config.IMAGE_DIR).exists())
    st.write("INDEX_DIR exists:", Path(config.INDEX_DIR).exists())
    st.write("Index exists:", Path(config.FIASS_INDEX_FILE).exists())
    st.write("Paths exists:", Path(config.IMAGE_PATH_FILE).exists())
    
    try:
        with open(config.IMAGE_PATH_FILE, "rb") as _f:
            _paths = _p.load(_f)
        st.write("image_paths.pkl count:", len(_paths))
        if _paths:
            _p0 = _paths[0]
            _abs0 = (Path(_p0) if Path(_p0).is_absolute() else Path(config.REPO_ROOT) / _p0)
            st.write("First path (stored):", _p0)
            st.write("First path (abs):", _abs0.as_posix())
            st.write("First path exists here:", _abs0.exists())
    except Exception as _e:
        st.error(f"Failed to read image_paths.pkl: {_e}")
        from pathlib import Path
st.write("Index path:", Path(config.FIASS_INDEX_FILE).as_posix())
st.write("manage_index contents:", [p.name for p in Path(config.INDEX_DIR).glob("*")])
try:
    st.write("Index size (bytes):", Path(config.FIASS_INDEX_FILE).stat().st_size)
except Exception:
    st.write("Index size (bytes): <not found>")



    # -------------------------------------------------------

    threshold = st.slider(
        "Minimum similarity (0–1)",
        0.0, 1.0, 0.30, 0.01,  # friendlier default so results aren't hidden
        help="Hide weak matches below this score"
    )

    st.markdown("<div class='cta-btn'>", unsafe_allow_html=True)
    find_clicked = st.button("✨ Find Similar", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if find_clicked:
        # Pick embedding priority: URL > Upload
        query_embedding = st.session_state.url_embedding if st.session_state.url_embedding is not None else uploaded_embedding

        if query_embedding is None:
            st.warning("Please upload an image or load one from URL first.")
        else:
            with st.spinner("Searching the index…"):
                try:
                    results = search_similar(query_embedding, config.TOP_K)
                except Exception as e:
                    results = []
                    st.error(f"Search failed: {e}")

            # Accept both formats: [(path, score)] or [path, ...]
            def iter_results(rs):
                for r in rs:
                    if isinstance(r, (list, tuple)) and len(r) == 2:
                        yield r[0], float(r[1])
                    else:
                        yield r, None

            st.write(f"**Top {config.TOP_K} results (≥ {threshold:.2f})**")
            cols = st.columns(3, gap="medium")
            shown = 0
            for idx, (path, score) in enumerate(iter_results(results)):
                ui_path = _resolve_ui_path(path)
                if not os.path.exists(ui_path):
                    # temporary: show which files are missing
                    st.write("Missing file (debug):", ui_path)
                    continue
                if score is None or score >= threshold:
                    with cols[idx % 3]:
                        show_image(
                            ui_path,
                            caption=(f"Similarity: {score:.3f}" if score is not None else None),
                        )
                    shown += 1

            if shown == 0:
                st.warning("No similar images above the threshold (or files missing).")

    st.markdown("</div>", unsafe_allow_html=True)

# -------- Optional: clean up temp uploaded preview (upload tab) so the folder doesn't grow
# (We keep URL images in tmp to allow re-search; Clear URL removes it.)
if 'uploaded_preview_path' in locals() and uploaded_preview_path:
    try:
        remove_uploaded_image(uploaded_preview_path)
    except Exception:
        pass
