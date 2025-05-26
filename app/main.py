from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import uuid

# ─── Paths Setup ───────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, 'static', 'results')
MODEL_PATH = os.path.join(BASE_DIR, '..', 'model', 'pet_detector_colab32', 'runs', 'detect', 'pet_detector', 'weights',
                          'best.pt')

# ─── Flask App Setup ───────────────────────────────────
app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# ─── Create Folders ────────────────────────────────────
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ─── Load YOLO Model ───────────────────────────────────
model = YOLO(MODEL_PATH)


# ─── Routes ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4().hex}_{filename}"
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    result_path = os.path.join(app.config['RESULT_FOLDER'], unique_filename)

    # Save uploaded image
    file.save(image_path)

    # Run YOLO detection
    results = model.predict(image_path, conf=0.5, save=False)
    image = cv2.imread(image_path)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model.names[int(box.cls[0])]
        conf = box.conf[0]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(result_path, image)

    return render_template('index.html', result_image=f'results/{unique_filename}')


# ─── Run ────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)
