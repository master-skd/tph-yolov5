import argparse
import sys
import os
import cv2
import random
import yaml
import shutil
import time
from pathlib import Path
from tqdm import tqdm

# Add current directory to path to import train
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import train

def generate_online_dataset(video_path, object_dir, output_dir, num_frames=None, objects_per_frame=3):
    output_dir = Path(output_dir)
    img_dir = output_dir / 'images'
    lbl_dir = output_dir / 'labels'
    
    if output_dir.exists():
        shutil.rmtree(output_dir)
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    
    # Load objects
    object_dir = Path(object_dir)
    if not object_dir.exists():
        print(f"Object directory {object_dir} does not exist!")
        return None
        
    objects = {} # cls -> [path, ...]
    for cls_dir in object_dir.iterdir():
        if cls_dir.is_dir() and cls_dir.name.isdigit():
            cls = int(cls_dir.name)
            objs = list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.png'))
            if objs:
                objects[cls] = objs
            
    if not objects:
        print("No extracted objects found!")
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return None
        
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frames and num_frames < frame_count:
        frame_count = num_frames
        
    print(f"Generating dataset from video: {video_path} ({frame_count} frames)")
    
    # Pre-calculate classes for yaml
    nc = max(objects.keys()) + 1
    
    for i in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break
            
        h, w = frame.shape[:2]
        labels = []
        
        # Paste objects
        for _ in range(objects_per_frame):
            # Pick random class
            cls = random.choice(list(objects.keys()))
            obj_path = random.choice(objects[cls])
            obj_img = cv2.imread(str(obj_path))
            
            if obj_img is None:
                continue
                
            oh, ow = obj_img.shape[:2]
            
            # Resize object if too big (max 50% of screen dimension)
            max_h, max_w = h * 0.5, w * 0.5
            scale = 1.0
            if oh > max_h or ow > max_w:
                scale = min(max_h/oh, max_w/ow)
            
            # Random scale variation (0.5 to 1.5 of original/fitted size)
            scale *= random.uniform(0.5, 1.5)
            
            if scale != 1.0:
                obj_img = cv2.resize(obj_img, (0,0), fx=scale, fy=scale)
                oh, ow = obj_img.shape[:2]
            
            # Ensure it fits
            if ow >= w or oh >= h:
                continue
                
            rx = random.randint(0, w - ow)
            ry = random.randint(0, h - oh)
            
            # Paste (simple overwrite)
            frame[ry:ry+oh, rx:rx+ow] = obj_img
            
            # Label: cls xc yc w h (normalized)
            xc = (rx + ow/2) / w
            yc = (ry + oh/2) / h
            nw = ow / w
            nh = oh / h
            labels.append(f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
            
        # Save
        img_name = f"frame_{i:06d}.jpg"
        cv2.imwrite(str(img_dir / img_name), frame)
        with open(lbl_dir / (img_name.replace('.jpg', '.txt')), 'w') as f:
            f.write('\n'.join(labels))
            
    cap.release()
    
    # Create dataset.yaml
    # Note: train.py expects absolute paths usually
    yaml_content = {
        'path': str(output_dir.absolute()),
        'train': 'images',
        'val': 'images', # Use same for val
        'nc': nc,
        'names': [f'class{i}' for i in range(nc)] 
    }
    
    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(yaml_content, f)
        
    return str(yaml_path)

def run():
    parser = argparse.ArgumentParser()
    # Online training specific args
    parser.add_argument('--video-path', type=str, required=True, help='Path to video file')
    parser.add_argument('--object-dir', type=str, required=True, help='Path to extracted objects directory')
    parser.add_argument('--output-dir', type=str, default='temp_online_data', help='Temporary dataset directory')
    parser.add_argument('--frames', type=int, default=None, help='Number of frames to use (default: all)')
    parser.add_argument('--objs-per-frame', type=int, default=3, help='Number of objects to paste per frame')
    
    # Standard train args
    parser.add_argument('--weights', type=str, default='yolov5s.pt')
    parser.add_argument('--freeze', type=str, default='0', help='Layers to freeze. Int or list (e.g. "0,1,10")')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='online_exp', help='save to project/name')
    
    # Parse known args to allow other train.py args if needed (though we only map specific ones here)
    opt, unknown = parser.parse_known_args()
    
    # 1. Generate Dataset
    print("Generating online dataset...")
    dataset_yaml = generate_online_dataset(
        opt.video_path, 
        opt.object_dir, 
        opt.output_dir, 
        opt.frames, 
        opt.objs_per_frame
    )
    
    if not dataset_yaml:
        print("Dataset generation failed.")
        return

    # 2. Setup Train Options
    # We load default train options and override
    train_opt = train.parse_opt(known=True)
    
    # Override
    train_opt.data = dataset_yaml
    train_opt.weights = opt.weights
    train_opt.epochs = opt.epochs
    train_opt.batch_size = opt.batch_size
    train_opt.imgsz = opt.imgsz
    train_opt.project = opt.project
    train_opt.name = opt.name
    
    # Handle Freeze
    # Check if comma separated list
    if ',' in str(opt.freeze):
        train_opt.freeze = [int(x) for x in opt.freeze.split(',')]
    else:
        try:
            train_opt.freeze = int(opt.freeze)
        except ValueError:
            print("Invalid freeze argument. Must be int or comma-separated ints.")
            return

    # 3. Run Training
    print(f"Starting training with data={dataset_yaml}, freeze={train_opt.freeze}")
    train.main(train_opt)

if __name__ == "__main__":
    run()