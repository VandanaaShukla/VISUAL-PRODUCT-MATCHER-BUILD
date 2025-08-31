import os
import uuid
from core import config

# Save uploads in a temp folder under the index dir, NOT in images/
TMP_DIR = os.path.join(config.INDEX_DIR, "tmp_uploads")
os.makedirs(TMP_DIR, exist_ok=True)

def save_uploaded_image(uploaded_file):
    _, ext = os.path.splitext(uploaded_file.name)
    unique_filename = f"{uuid.uuid4()}{ext}"
    image_path = os.path.join(TMP_DIR, unique_filename)
    with open(image_path, "wb") as buffer:
        buffer.write(uploaded_file.read())
    return image_path

def remove_uploaded_image(image_path):
    if os.path.exists(image_path):
        try:
            os.remove(image_path)
        except Exception:
            pass
