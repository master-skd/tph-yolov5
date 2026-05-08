import argparse
import os
import cv2
import yaml
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm

def extract_objects(data_yaml, save_dir):
    # Load data yaml
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    
    # Get train path
    train_path = data.get('train')
    if not train_path:
        print("Error: 'train' path not found in yaml")
        return

    # Resolve paths
    data_path = data.get('path', '.')
    if isinstance(train_path, str):
        train_path = [train_path]
    
    img_paths = []
    for p in train_path:
        p = Path(p)
        if not p.is_absolute():
            # Try relative to the 'path' in yaml if specified
            p = Path(data_path) / p
            
        if p.is_file():
            with open(p, 'r') as f:
                img_paths.extend([os.path.join(p.parent, x.strip()) for x in f.readlines()])
        elif p.is_dir():
            for root, dirs, files in os.walk(p):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif')):
                        img_paths.append(os.path.join(root, file))
    
    save_dir = Path(save_dir)
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Found {len(img_paths)} images. Extracting objects...")
    
    count = 0
    for img_path in tqdm(img_paths):
        img_path = Path(img_path)
        if not img_path.exists():
            continue
            
        # Infer label path
        # 1. images/train/im0.jpg -> labels/train/im0.txt
        # 2. images/im0.jpg -> labels/im0.txt
        
        # Check standard YOLOv5 structure
        # ../images/.. -> ../labels/..
        try:
            str_path = str(img_path)
            if '/images/' in str_path:
                label_path = str_path.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
            else:
                # Fallback: same dir or parallel 'labels' dir
                label_path = str(img_path.parent / 'labels' / (img_path.stem + '.txt'))
                if not os.path.exists(label_path):
                     label_path = str(img_path.parent.parent / 'labels' / (img_path.stem + '.txt'))
        except Exception:
            continue
            
        if not os.path.exists(label_path):
            continue
            
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls = int(parts[0])
                xc, yc, ww, hh = map(float, parts[1:5])
            except ValueError:
                continue
            
            # Convert to xyxy
            x1 = int((xc - ww/2) * w)
            y1 = int((yc - hh/2) * h)
            x2 = int((xc + ww/2) * w)
            y2 = int((yc + hh/2) * h)
            
            # Clip
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            crop = img[y1:y2, x1:x2]
            
            # Save
            cls_dir = save_dir / str(cls)
            cls_dir.mkdir(exist_ok=True)
            save_name = f"{img_path.stem}_{i}.jpg"
            cv2.imwrite(str(cls_dir / save_name), crop)
            count += 1
            
    print(f"Extracted {count} objects to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='data.yaml path')
    parser.add_argument('--save-dir', type=str, default='extracted_objects', help='output directory')
    args = parser.parse_args()
    
    extract_objects(args.data, args.save_dir)