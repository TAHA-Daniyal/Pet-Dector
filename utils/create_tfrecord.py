import os
import io
import pandas as pd
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

# Monkey-patch for TF2.x
# so label_map_util can use tf.gfile
try:
    tf.gfile = tf.io.gfile
except:
    pass

# Paths
after = os.path.dirname(__file__)
BASE_DIR       = os.path.dirname(__file__)
PROC_DIR       = os.path.join(BASE_DIR, "..", "data", "processed")
CSV_FILES      = {
    'train': os.path.join(PROC_DIR, "train.csv"),
    'val':   os.path.join(PROC_DIR, "val.csv"),
    'test':  os.path.join(PROC_DIR, "test.csv")
}
RAW_IMG_DIR    = os.path.join(BASE_DIR, "..", "data", "raw", "oxford_pets", "images")
LABEL_MAP_PATH = os.path.join(PROC_DIR, "pet_label_map.pbtxt")
OUTPUT_DIR     = os.path.join(PROC_DIR, "tfrecords")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load label map
label_map = label_map_util.get_label_map_dict(LABEL_MAP_PATH)


def create_tf_example(group, label_map):
    """
    group: DataFrame of all rows for a single image
    """
    filename = group['filename'].iloc[0]
    img_path = os.path.join(RAW_IMG_DIR, filename)
    with tf.io.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    image = Image.open(io.BytesIO(encoded_jpg))
    width, height = image.size

    # Image-level features
    feature_dict = {
        'image/height':    dataset_util.int64_feature(height),
        'image/width':     dataset_util.int64_feature(width),
        'image/filename':  dataset_util.bytes_feature(filename.encode('utf8')),
        'image/encoded':   dataset_util.bytes_feature(encoded_jpg),
        'image/format':    dataset_util.bytes_feature(b'jpg'),
    }

    # Initialize lists for bboxes
    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    classes_text, classes = [], []

    # Derive breed from the filename
    breed = os.path.splitext(filename)[0].split('_')[0].lower()

    # Populate lists for each object in the image
    for _, row in group.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(breed.encode('utf8'))
        classes.append(label_map[breed])

    # Add object-level features\
    feature_dict.update({
        'image/object/bbox/xmin':  dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax':  dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin':  dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax':  dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label':dataset_util.int64_list_feature(classes),
    })

    return tf.train.Example(features=tf.train.Features(feature=feature_dict))


def generate_tfrecord(split_name, csv_input):
    output_path = os.path.join(OUTPUT_DIR, f"{split_name}.record")
    df = pd.read_csv(csv_input)
    writer = tf.io.TFRecordWriter(output_path)
    grouped = df.groupby('filename')

    for filename, group in grouped:
        tf_example = create_tf_example(group, label_map)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print(f"✅ Wrote TFRecord for {split_name} ({output_path})")


if __name__ == "__main__":
    for split, csv_path in CSV_FILES.items():
        if os.path.exists(csv_path):
            generate_tfrecord(split, csv_path)
        else:
            print(f"⚠️  CSV not found: {csv_path}, skipping.")
