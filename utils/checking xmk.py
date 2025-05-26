# utils/check_mismatch.py

import os
import csv

# 1) Adjust these paths if your structure differs
BASE_DIR     = os.path.dirname(__file__)     # utils/
IMAGE_DIR    = os.path.join(BASE_DIR, '..', 'data', 'raw', 'oxford_pets', 'images')
XML_DIR      = os.path.join(BASE_DIR, '..', 'data', 'raw', 'oxford_pets', 'annotations', 'xmls')
OUTPUT_CSV   = os.path.join(BASE_DIR, '..', 'data', 'processed', 'mismatch_report.csv')

# 2) Gather basenames
image_bases = {os.path.splitext(f)[0] for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg','.png','jpeg'))}
xml_bases   = {os.path.splitext(f)[0] for f in os.listdir(XML_DIR)   if f.lower().endswith('.xml')}

# 3) Union of all basenames
all_bases = sorted(image_bases.union(xml_bases))

# 4) Write report
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
with open(OUTPUT_CSV, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['basename', 'has_image', 'has_xml'])
    for base in all_bases:
        writer.writerow([base, base in image_bases, base in xml_bases])

print(f"✅ Mismatch report written to {OUTPUT_CSV}")
