import os, shutil, random, pandas as pd

# 1. Paths
RAW_IMG_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "oxford_pets", "images")
CSV_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "converted_annotations.csv")

PROC_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
TRAIN_IMG  = os.path.join(PROC_DIR, "train")
VAL_IMG    = os.path.join(PROC_DIR, "val")
TEST_IMG   = os.path.join(PROC_DIR, "test")

# 2. Create output dirs
for d in (TRAIN_IMG, VAL_IMG, TEST_IMG):
    os.makedirs(d, exist_ok=True)

# 3. Read annotations
df = pd.read_csv(CSV_PATH)

# 4. Get unique filenames
all_files = df['filename'].unique().tolist()
random.shuffle(all_files)

# 5. Split ratios
n = len(all_files)
n_train = int(0.7 * n)
n_val   = int(0.15 * n)

train_files = set(all_files[:n_train])
val_files   = set(all_files[n_train:n_train + n_val])
test_files  = set(all_files[n_train + n_val:])

# 6. Helper: copy image
def cp(f, dest_dir):
    src = os.path.join(RAW_IMG_DIR, f)
    dst = os.path.join(dest_dir, f)
    if not os.path.exists(dst):
        shutil.copy(src, dst)

# 7. Distribute images
for fname in train_files:
    cp(fname, TRAIN_IMG)
for fname in val_files:
    cp(fname, VAL_IMG)
for fname in test_files:
    cp(fname, TEST_IMG)

# 8. Write per-split CSVs
for split_name, file_set in zip(
    ["train", "val", "test"],
    [train_files, val_files, test_files]
):
    out_csv = os.path.join(PROC_DIR, f"{split_name}.csv")
    df_split = df[df['filename'].isin(file_set)]
    df_split.to_csv(out_csv, index=False)
    print(f"→ Wrote {len(df_split)} annotations to {out_csv}")

print("✅ Dataset split complete.")
