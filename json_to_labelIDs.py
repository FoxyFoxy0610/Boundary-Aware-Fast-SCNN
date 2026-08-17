import json
import numpy as np
from PIL import Image, ImageDraw
import glob
import argparse
import os

def json_to_labelid_png(json_file, output_png, label_map):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    width = data['imageWidth']
    height = data['imageHeight']

    # 7 is usually the default ignore/unlabeled class in this context
    label_image = np.ones((height, width), dtype=np.uint8) * 7
    label_image_array = Image.fromarray(label_image)
    draw = ImageDraw.Draw(label_image_array)
    
    for shape in data['shapes']:
        label = shape['label']
        points = shape['points']

        if label in label_map:
            labelid = label_map[label]
        else:
            continue
            
        polygon = np.array(points, dtype=np.int32)
        polygon = [tuple(point) for point in polygon]
        draw.polygon(polygon, outline=labelid, fill=labelid)
    
    label_image_array.save(output_png)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert LabelMe JSON annotations to LabelID PNG for training.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing JSON files')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save PNG files (defaults to input_dir)')
    parser.add_argument('--dataset_type', type=str, default='custom_agricultural', choices=['custom_agricultural', 'asparagus'], help='Dataset type to determine label mapping')
    args = parser.parse_args()

    if args.dataset_type == 'custom_agricultural':
        label_map = {"lane": 20}
    elif args.dataset_type == 'asparagus':
        label_map = {"lane": 20, "plot": 25}

    out_dir = args.output_dir if args.output_dir else args.input_dir
    os.makedirs(out_dir, exist_ok=True)

    json_files = glob.glob(os.path.join(args.input_dir, '*.json'))
    if not json_files:
        print(f"No JSON files found in {args.input_dir}")
        
    for json_original in json_files:
        filename = os.path.basename(json_original)
        out_name = filename.replace('.json', '_labelIds.png')
        out_path = os.path.join(out_dir, out_name)
        
        print(f"Converting {filename} -> {out_name}")
        json_to_labelid_png(json_original, out_path, label_map)
