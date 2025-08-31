import os
import pandas as pd

data = []
id_counter = 1

for category in os.listdir("images"):
    folder = os.path.join("images", category)
    if os.path.isdir(folder):  # make sure it's a folder
        for img in os.listdir(folder):
            if img.lower().endswith((".jpg", ".jpeg", ".png")):
                data.append({
                    "id": id_counter,
                    "name": f"{category} - {id_counter}",  # product name
                    "category": category,                  # folder name
                    "image_path": os.path.join(folder, img) # relative path
                })
                id_counter += 1

df = pd.DataFrame(data)
df.to_csv("products.csv", index=False)
print(f"✅ products.csv created with {len(df)} items")
