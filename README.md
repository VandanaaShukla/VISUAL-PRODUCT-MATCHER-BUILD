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
