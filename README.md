# Visual Product Matcher Build

This is a simple image similarity search tool to find visually similar images based on similarity thresholds. The app is built with the help of following: [Streamlit](https://streamlit.io/), [OpenAI CLIP](https://github.com/openai/CLIP), and **scikit-learn**.  

**features of the app** include searching images via uploads and URL. On uploading the image or pasting the url, the image is displayed.
when clicked on **find similar** we are able to find images based on similarity threshold.

**the dataset** contains 250-270 images in the category of  ["Nike Shoes","Adidas Shoes","Puma Shoes","Handbags","Watches",
    "Backpacks","Sunglasses","Jackets","T-shirts","Dresses",
    "Laptops","Smartphones","Headphones","Chairs","Tables",
    "Dogs","Cats","Birds","Cars","Bikes",
    "Plants","Flowers","Paintings","Sculptures","Cups"]

    The **similarity score** threshold is set to **0.60** by default and it return **TOP -3** visually similar images.



**APPROACH**
The project implements a Visual Product Matcher using a combination of OpenAI CLIP embeddings and scikit-learn similarity search. The core idea is to represent every image in the dataset as a high-dimensional embedding and then compare new queries (uploaded images or URLs) against these embeddings to retrieve the most similar items.
First, a dataset of ~250 product images (shoes, handbags, watches, etc.) was prepared, with metadata stored in products.csv. Each image was passed through CLIP’s ViT-B/32 model to extract a 512-dimensional embedding vector. These embeddings were stored in an index file (index_file.pkl), along with image paths (image_paths.pkl).
For similarity search, we used Nearest Neighbors (cosine similarity) from scikit-learn. At query time, the uploaded image (or URL image) is embedded via CLIP, and its similarity with dataset embeddings is computed. Results are ranked by similarity scores and displayed in a responsive Streamlit interface, with a slider to filter weak matches.
The app supports:
Uploading an image or fetching from a URL
Viewing the uploaded image
Displaying top-k (k=3) similar items with similarity scores

Finally, the project was deployed on Streamlit Community Cloud for public access.





---

## 🚀 Features

- **Image Upload / URL Input**: 
  - Upload a JPG/PNG from your device
  - Or paste an image URL to fetch directly
- **Search Interface**:
  - View your uploaded image
  - Get top-k visually similar items
  - Filter results by **similarity score threshold**
- **Product Database**:
  - 250+ product images (shoes, handbags, watches, etc.)
  - Each product includes **ID, name, category, and image path**
- **Responsive UI**: Works on desktop & mobile
- 

---

## 🛠️ Tech Stack

- **Backend / Model**:
  - [OpenAI CLIP](https://github.com/openai/CLIP) (ViT-B/32) for embeddings
  - `scikit-learn` `NearestNeighbors` for similarity search
- **Frontend**:
  - [Streamlit](https://streamlit.io/) for fast UI
- **Data**:
  - `products.csv` – metadata (ID, name, category, path)
  - `image_paths.pkl` + `index_file.pkl` – precomputed embeddings
- **Hosting**:
  - Streamlit Community Cloud (free)

---

## 📂 Project Structure

visual-product-matcher-build/
│
├── core/
│ ├── clip_utils.py # CLIP model + embedding functions
│ ├── index_manager.py # Index handling with scikit-learn
│ ├── storage.py # Save/remove uploaded images
│ └── config.py # Config paths, constants
│
├── images/ # Dataset images (250+ products)
├── index_manager/
│ ├── image_paths.pkl # List of image paths
│ └── index_file.pkl # Embeddings array
│
├── products.csv # Product metadata
├── streamlit_ui/
│ └── app.py # Main Streamlit app
├── requirements.txt # Dependencies
└── README.md # Project documentation


---

## ▶️ Run Locally

Clone the repo:
```bash
git clone https://github.com/VandanaaShukla/VISUAL-PRODUCT-MATCHER-BUILD.git
```

create a virtual environment 
```bash
python -m venv .venv
```
Activate the virtual environment
```bash
source .venv/bin/activate    # Mac/Linux
.venv\Scripts\activate       # Windows
```
Install all the dependencies
```bash
pip install -r requirements.txt
```
Command to run the app 
```bash
streamlit run streamlit_ui/app.py
```
