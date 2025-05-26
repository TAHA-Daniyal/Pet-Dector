import os
import pandas as pd

# 1. Paths
BASE       = os.path.dirname(__file__)
PROC       = os.path.join(BASE, "..", "data", "processed")
# Read images directly from your processed folder:
IMG_SRC    = PROC                    # contains `train/` and `val/`
IMG_DST    = os.path.join(BASE, "..", "data", "images")
LBL_DST    = os.path.join(BASE, "..", "data", "labels")

# Ensure output dirs exist
for split in ("train", "val"):
    os.makedirs(os.path.join(IMG_DST, split), exist_ok=True)
    os.makedirs(os.path.join(LBL_DST, split), exist_ok=True)

# 2. Read your split CSVs
df_train = pd.read_csv(os.path.join(PROC, "train.csv"))
df_val   = pd.read_csv(os.path.join(PROC, "val.csv"))
dfs      = {"train": df_train, "val": df_val}

# 3. Build breed→class_id map from train split
breeds = sorted({
    os.path.splitext(fname)[0].split('_')[0].lower()
    for fname in df_train["filename"]
})
class_map = {breed: idx for idx, breed in enumerate(breeds)}

# (Optional) write out a names file for reference
with open(os.path.join(PROC, "pet.names"), "w") as f:
    f.write("\n".join(breeds))

# 4. Process each split
for split, df in dfs.items():
    for _, row in df.iterrows():
        fname = row["filename"]
        breed = os.path.splitext(fname)[0].split('_')[0].lower()
        cls_id = class_map[breed]

        # Copy or link the image
        src_img = os.path.join(IMG_SRC, split, fname)          # e.g. data/processed/train/Abyssinian_1.jpg
        dst_img = os.path.join(IMG_DST, split, fname)          # e.g. data/images/train/Abyssinian_1.jpg
        if not os.path.exists(dst_img):
            os.link(src_img, dst_img)

        # Compute normalized bbox coordinates
        w, h = row["width"], row["height"]
        x_c = (row["xmin"] + row["xmax"]) / 2 / w
        y_c = (row["ymin"] + row["ymax"]) / 2 / h
        bw  = (row["xmax"] - row["xmin"]) / w
        bh  = (row["ymax"] - row["ymin"]) / h

        # Write YOLO label file
        lbl_path = os.path.join(LBL_DST, split, fname.replace(".jpg", ".txt"))
        with open(lbl_path, "a") as f:
            f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")

print("✅ Converted processed dataset into YOLO format:")
print("   • Images: data/images/{train,val}")
print("   • Labels: data/labels/{train,val}")
