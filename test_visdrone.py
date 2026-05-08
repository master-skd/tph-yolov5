# check_mapping.py
import json
import os

# 加载annotation文件
ann_file = '../datasets/VisDrone/VisDrone2019-DET/annotations/val.json'
with open(ann_file, 'r') as f:
    data = json.load(f)

# 创建文件名到ID的映射
img_name_to_id = {}
for img_info in data['images']:
    # 获取纯文件名（不带路径）
    filename = os.path.basename(img_info['file_name'])
    img_name_to_id[filename] = img_info['id']

print(f"Total images in annotation: {len(img_name_to_id)}")
print(f"Total annotations: {len(data['annotations'])}")

# 检查实际的图片文件
image_dir = '../datasets/VisDrone/VisDrone2019-DET/val/images'
actual_images = os.listdir(image_dir)
print(f"\nActual images in directory: {len(actual_images)}")

# 找出在annotation中不存在的图片
missing_in_ann = []
for img in actual_images:
    if img not in img_name_to_id:
        missing_in_ann.append(img)

print(f"\nImages missing in annotation: {len(missing_in_ann)}")
if missing_in_ann:
    print("First 10 missing images:")
    for img in missing_in_ann[:10]:
        print(f"  {img}")
    
    # 检查出错的具体图片
    error_img = "0000001_02999_d_0000005.jpg"
    print(f"\nChecking error image '{error_img}':")
    if error_img in actual_images:
        print(f"  ✓ Image exists in directory")
    if error_img in img_name_to_id:
        print(f"  ✓ Image found in annotation (ID: {img_name_to_id[error_img]})")
    else:
        print(f"  ✗ Image NOT found in annotation mapping")
        
    # 检查相似的文件名（可能有命名差异）
    print(f"\nSearching for similar filenames in annotation:")
    similar = [name for name in img_name_to_id.keys() if "0000001_02999" in name]
    for name in similar[:5]:
        print(f"  {name} -> ID: {img_name_to_id[name]}")