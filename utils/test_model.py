import os
from ultralytics import YOLO
import cv2

# ─── Config ────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'model',
    'pet_detector_colab32','runs','detect','pet_detector', 'weights', 'best.pt'
)
#TEST_IMG_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'test')
TEST_IMG_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'test' )
# Save annotated results alongside processed data
OUTPUT_DIR     = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'test_results')
CONFIDENCE_THR = 0.5

# Colors (B, G, R)
BOX_COLOR       = (0, 165, 255)    # orange
TEXT_COLOR      = (255, 255, 255)  # white text
BG_COLOR        = (0, 0, 0)        # black background for text
FONT            = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE      = 0.6
BOX_THICKNESS   = 2
TEXT_THICKNESS  = 1  # thinner text for clarity

# ────────────────────────────────────────────────────────────────────────────

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def annotate_and_save(model, img_path, out_dir):
    img = cv2.imread(img_path)
    results = model.predict(source=img_path, conf=CONFIDENCE_THR, save=False)
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = box.conf[0]

        def draw_label(img, label, x1, y1, x2, y2):
            font_scale = 0.5
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = 1
            color = (0, 255, 0)

            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Calculate text size
            (w, h), _ = cv2.getTextSize(label, font, font_scale, thickness)

            # Adjust label position if it’s going off the image top
            text_y = y1 - 10 if y1 - 10 > h else y1 + h + 10

            # Draw background rectangle for label
            cv2.rectangle(img, (x1, text_y - h), (x1 + w, text_y), color, -1)

            # Put label text
            cv2.putText(img, label, (x1, text_y - 2), font, font_scale, (0, 0, 0), thickness)
    # Save result
    fname = os.path.basename(img_path)
    out_path = os.path.join(out_dir, fname)
    cv2.imwrite(out_path, img)


def main():
    print(f"Loading model from {MODEL_PATH} …")
    model = YOLO(MODEL_PATH)

    ensure_dir(OUTPUT_DIR)
    print(f"Writing annotated images to {OUTPUT_DIR}")

    for fname in os.listdir(TEST_IMG_DIR):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(TEST_IMG_DIR, fname)
        print(f"Processing {fname} …")
        annotate_and_save(model, img_path, OUTPUT_DIR)

    print("Done. Check your results folder.")

if __name__ == "__main__":
    main()
