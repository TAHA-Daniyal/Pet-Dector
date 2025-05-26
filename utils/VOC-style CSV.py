import pandas as pd

df = pd.read_csv("makesense_annotations.csv")

df['xmin'] = df['bbox_x']
df['ymin'] = df['bbox_y']
df['xmax'] = df['bbox_x'] + df['bbox_width']
df['ymax'] = df['bbox_y'] + df['bbox_height']
df['filename'] = df['image_name']
df['width'] = df['image_width']
df['height'] = df['image_height']
df['class'] = df['label_name']

final_df = df[['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']]
final_df.to_csv("converted_annotations.csv", index=False)
