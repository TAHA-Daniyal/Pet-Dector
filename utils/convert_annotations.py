import os
import glob
import csv
import xml.etree.ElementTree as ET

# 1. Paths
ANNOTATIONS_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "data", "raw", "oxford_pets", "annotations", "xmls"
)
IMAGES_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "data", "raw", "oxford_pets", "images"
)
OUTPUT_CSV = os.path.join(
    os.path.dirname(__file__),
    "..", "data", "processed", "annotations.csv"
)

# 2. Prepare CSV writer
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
csvfile = open(OUTPUT_CSV, "w", newline="")
writer = csv.writer(csvfile)
writer.writerow(["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"])

# 3. Loop through each XML file
for xml_path in glob.glob(os.path.join(ANNOTATIONS_DIR, "*.xml")):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    fname = root.find("filename").text
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    # 4. For each object (pet) in the image
    for obj in root.findall("object"):
        cls = obj.find("name").text  # e.g. "cat" or "dog" or breed name
        bnd = obj.find("bndbox")
        xmin = int(bnd.find("xmin").text)
        ymin = int(bnd.find("ymin").text)
        xmax = int(bnd.find("xmax").text)
        ymax = int(bnd.find("ymax").text)

        # 5. Write one row per box
        writer.writerow([fname, width, height, cls, xmin, ymin, xmax, ymax])

csvfile.close()
print(f"✅ Wrote annotations to {OUTPUT_CSV}")
