import torch
import yaml
from pathlib import Path
from tqdm import tqdm
import sys
import os

# 将当前目录添加到路径，防止找不到utils
sys.path.append(os.getcwd())

from utils.general import box_iou, check_img_size, non_max_suppression, xywh2xyxy
from utils.datasets import LoadImagesAndLabels
from utils.torch_utils import select_device

# ================= 配置参数 =================
CONF_THRES = 0.45  # 置信度阈值
IOU_THRES = 0.45   # NMS 和 TP 判定阈值
imgsz_target = 640 # 期望输入尺寸

weights = "/data1/code/xy/tph-yolov5/runs/train/tphv5s-antiuav-cop-api-dconv/weights/best.pt"
data_yaml = "data/anti-uav.yaml"
# ===========================================

def run_test():
    # 1. 加载模型
    device = select_device('cuda:2')
    print(f"Loading model from {weights}...")
    ckpt = torch.load(weights, map_location=device)
    model = ckpt['model'].float().eval().to(device)
    
    # 2. 修正图片尺寸 (解决报错的关键)
    stride = int(model.stride.max())  # 获取模型最大步长 (通常32)
    imgsz = check_img_size(imgsz_target, s=stride) 
    if imgsz != imgsz_target:
        print(f"⚠️ Warning: --img-size {imgsz_target} must be multiple of max stride {stride}, updating to {imgsz}")

    # 3. 加载数据集
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    
    # 构造验证集路径
    # 注意：根据你的yaml结构，这里可能需要调整。假设 path 和 test 拼接是正确路径
    test_path = str(Path(data.get("path", "")) / data["test"])
    if not os.path.exists(test_path):
        # 兼容绝对路径的情况
        test_path = data["test"]

    print(f"Loading dataset from {test_path}...")
    dataset = LoadImagesAndLabels(
        path=test_path,
        img_size=imgsz,
        batch_size=1,
        augment=False,
        rect=False,       # 验证时建议关闭矩形推理以保持尺寸一致
        stride=stride
    )

    tp = 0
    fp = 0
    total_gt = 0

    print("Start inference...")
    
    # 4. 推理循环
    for img, targets, paths, shapes in tqdm(dataset):
        img = img.to(device).float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # ---- Forward ----
        with torch.no_grad():
            preds = model(img)[0]
            # ---- NMS (必须步骤) ----
            # 过滤低置信度、去除重叠框
            preds = non_max_suppression(preds, CONF_THRES, IOU_THRES)

        # 批次大小为1，直接取第一个结果
        det = preds[0] 

        # ---- 处理 Ground Truth ----
        # targets shape: [num_obj, 6] -> [batch_idx, class, x, y, w, h]
        gt_boxes = targets[:, 2:6].to(device) * imgsz
        
        # 统计 GT 总数 (用于后续计算 Recall，虽然这里只需要 FPR)
        total_gt += len(gt_boxes)

        # 如果没有 GT 且没有预测框 -> 跳过
        if len(gt_boxes) == 0 and (det is None or len(det) == 0):
            continue
            
        # 如果没有 GT 但有预测框 -> 全部是 FP
        if len(gt_boxes) == 0 and len(det) > 0:
            fp += len(det)
            continue
            
        # 如果有 GT 但没有预测框 -> 只有 FN (此处不计入TP/FP)
        if len(gt_boxes) > 0 and (det is None or len(det) == 0):
            continue

        # ---- 匹配计算 ----
        # 将 GT 从 xywh (中心) 转为 xyxy (角点)
        gt_boxes = xywh2xyxy(gt_boxes)
        pred_boxes = det[:, :4]

        # 计算 IoU矩阵 [预测数量, GT数量]
        ious = box_iou(pred_boxes, gt_boxes)

        matched_gt = set()
        
        # 遍历每个预测框
        for i in range(len(pred_boxes)):
            if len(gt_boxes) > 0:
                # 找到该预测框匹配度最高的 GT
                max_iou, gt_idx = ious[i].max(0)
                
                # 判定条件：IoU 达标 且 该 GT 未被占用
                if max_iou >= IOU_THRES and gt_idx.item() not in matched_gt:
                    tp += 1
                    matched_gt.add(gt_idx.item())
                else:
                    fp += 1
            else:
                fp += 1

    # 5. 输出结果
    print("-" * 30)
    print(f"Total Images: {len(dataset)}")
    print(f"TP (True Positives): {tp}")
    print(f"FP (False Positives): {fp}")
    
    denom = tp + fp
    if denom == 0:
        print("No detections made.")
    else:
        # 误检率 (False Discovery Rate) = FP / (TP + FP)
        fdr = fp / denom
        print(f"Precision (TP / (TP + FP)): {tp / denom:.4f}")
        print(f"False Positive Rate (FP / (TP + FP)): {fdr:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    run_test()