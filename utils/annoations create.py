from ultralytics import YOLO
import os, xml.etree.ElementTree as ET

model = YOLO('models/pet_detector_colab32/weights/best.pt')
img_dir = 'data/raw/oxford_pets/images'
xml_dir = 'data/raw/oxford_pets/annotations/xmls'
os.makedirs(xml_dir, exist_ok=True)

for img_file in os.listdir(img_dir):
    if not img_file.endswith('.jpg'): continue
    img_path = os.path.join(img_dir, img_file)
    results = model.predict(source=img_path, conf=0.5)[0]
    # Build a basic VOC XML
    root = ET.Element('annotation')
    ET.SubElement(root, 'filename').text = img_file
    size = ET.SubElement(root, 'size')
    width, height = results.orig_shape[1], results.orig_shape[0]
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = '3'
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls   = model.names[int(box.cls[0])]
        obj   = ET.SubElement(root, 'object')
        ET.SubElement(obj, 'name').text = cls
        bnd   = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bnd, 'xmin').text = str(x1)
        ET.SubElement(bnd, 'ymin').text = str(y1)
        ET.SubElement(bnd, 'xmax').text = str(x2)
        ET.SubElement(bnd, 'ymax').text = str(y2)
    tree = ET.ElementTree(root)
    xml_path = os.path.join(xml_dir, img_file.replace('.jpg', '.xml'))
    tree.write(xml_path)
