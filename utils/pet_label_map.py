import os
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(__file__)
PROC_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
CSV_PATH = os.path.join(PROC_DIR, "train.csv")   # use train.csv to get all breeds
OUT_PATH = os.path.join(PROC_DIR, "pet_label_map.pbtxt")

# 1) Read train.csv
df = pd.read_csv(CSV_PATH)

# 2) Derive breed from filenames, e.g. "British_Shorthair_10.jpg" → "british"
breeds = sorted({
    os.path.splitext(fname)[0].split('_')[0].lower()
    for fname in df['filename']
})

# 3) Write pbtxt
with open(OUT_PATH, "w") as f:
    for idx, breed in enumerate(breeds, start=1):
        f.write("item {\n")
        f.write(f"  id: {idx}\n")
        f.write(f"  name: '{breed}'\n")
        f.write("}\n\n")

print(f"✅ Wrote label map with {len(breeds)} items to {OUT_PATH}")
