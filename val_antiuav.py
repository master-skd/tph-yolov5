import os
import json
import time
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, box_iou
from utils.torch_utils import select_device


def compute_ap(recall, precision):
    """
    计算 AP，采用常见的 precision envelope 方式
    recall, precision: 1D numpy arrays
    """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # precision envelope
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # 积分
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap


def evaluate_map(predictions, num_gt, iou_threshold=0.5):
    """
    predictions: list of dict
        each item = {
            'confidence': float,
            'iou': float,
            'matched': bool   # 对于该 IoU 阈值是否匹配成功
        }
    num_gt: 总 GT 数
    """
    if len(predictions) == 0:
        return 0.0, np.array([]), np.array([])

    predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)

    tp = np.array([1 if p['matched'] and p['iou'] >= iou_threshold else 0 for p in predictions])
    fp = np.array([0 if p['matched'] and p['iou'] >= iou_threshold else 1 for p in predictions])

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recall = tp_cum / (num_gt + 1e-16)
    precision = tp_cum / (tp_cum + fp_cum + 1e-16)

    ap = compute_ap(recall, precision)
    return ap, precision, recall


def evaluate_antiuav_yolo_metrics(weights, data_dir, imgsz=640, conf_thres=0.001, iou_thres_nms=0.45):
    device = select_device('')
    model = attempt_load(weights, map_location=device)
    model.eval()

    test_path = Path(data_dir) / 'test'
    video_folders = [f for f in test_path.iterdir() if f.is_dir()]

    iou_thresholds = np.arange(0.5, 0.96, 0.05)

    total_gt = 0
    total_frames = 0
    tp_50 = 0
    fp_50 = 0
    fn_50 = 0
    tn_50 = 0

    all_ious = []
    inference_times = []

    # 用于 mAP 计算：收集所有预测
    all_predictions_per_threshold = {t: [] for t in iou_thresholds}

    print(f"开始评测权重: {weights}")
    print(f"测试集路径: {test_path}")

    for folder in tqdm(video_folders):
        video_file = folder / 'visible.mp4'
        json_file = folder / 'visible.json'

        if not video_file.exists() or not json_file.exists():
            continue

        with open(json_file, 'r') as f:
            anno = json.load(f)

        gt_rects = anno['gt_rect']   # [x, y, w, h]
        exist = anno['exist']

        cap = cv2.VideoCapture(str(video_file))
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_idx >= len(exist):
                break

            total_frames += 1
            h, w = frame.shape[:2]

            # 预处理
            img = cv2.resize(frame, (imgsz, imgsz))
            img = img.transpose((2, 0, 1))[::-1]  # HWC->CHW, BGR->RGB
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device).float() / 255.0
            img = img.unsqueeze(0)

            # 推理计时
            t1 = time.time()
            with torch.no_grad():
                pred = model(img)[0]
            pred = non_max_suppression(pred, conf_thres, iou_thres_nms)[0]
            t2 = time.time()
            inference_times.append(t2 - t1)

            has_gt = bool(exist[frame_idx])

            if has_gt:
                total_gt += 1
                gt_box = gt_rects[frame_idx]
                gt_tensor = torch.tensor(
                    [[gt_box[0], gt_box[1], gt_box[0] + gt_box[2], gt_box[1] + gt_box[3]]],
                    dtype=torch.float32,
                    device=device
                )
            else:
                gt_tensor = None

            # 处理预测框
            frame_preds = []
            if pred is not None and len(pred) > 0:
                pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()

                for j in range(len(pred)):
                    box = pred[j:j+1, :4]
                    conf = float(pred[j, 4].item())

                    if has_gt:
                        iou = box_iou(gt_tensor, box).item()
                    else:
                        iou = 0.0

                    frame_preds.append({
                        'box': box,
                        'conf': conf,
                        'iou': iou
                    })

            # =========================
            # IoU / success rate / P R F1 @0.5
            # =========================
            if has_gt:
                if len(frame_preds) > 0:
                    best_idx = np.argmax([p['iou'] for p in frame_preds])
                    best_iou = frame_preds[best_idx]['iou']
                    all_ious.append(best_iou)

                    if best_iou >= 0.5:
                        tp_50 += 1
                        # 除最佳匹配外，其他预测框算 FP
                        fp_50 += max(0, len(frame_preds) - 1)
                    else:
                        fp_50 += len(frame_preds)
                        fn_50 += 1
                else:
                    all_ious.append(0.0)
                    fn_50 += 1
            else:
                # 没有 GT
                if len(frame_preds) > 0:
                    fp_50 += len(frame_preds)
                else:
                    tn_50 += 1

            # =========================
            # 为各 IoU 阈值准备 AP 数据
            # 单 GT 场景：每个阈值下最多一个预测记为 TP，其余为 FP
            # =========================
            for t in iou_thresholds:
                if has_gt:
                    if len(frame_preds) > 0:
                        best_idx = np.argmax([p['iou'] for p in frame_preds])
                        for k, p in enumerate(frame_preds):
                            matched = (k == best_idx and p['iou'] >= t)
                            all_predictions_per_threshold[t].append({
                                'confidence': p['conf'],
                                'iou': p['iou'],
                                'matched': matched
                            })
                    # 没有预测时，不往 predictions 里加，漏检由 num_gt 体现
                else:
                    # 无 GT，所有预测都是 FP
                    for p in frame_preds:
                        all_predictions_per_threshold[t].append({
                            'confidence': p['conf'],
                            'iou': 0.0,
                            'matched': False
                        })

            frame_idx += 1

        cap.release()

    # 基础指标
    precision_50 = tp_50 / (tp_50 + fp_50 + 1e-16)
    recall_50 = tp_50 / (tp_50 + fn_50 + 1e-16)
    f1_50 = 2 * precision_50 * recall_50 / (precision_50 + recall_50 + 1e-16)
    success_rate = tp_50 / (total_gt + 1e-16)
    mean_iou = np.mean(all_ious) if len(all_ious) > 0 else 0.0

    # AP / mAP
    ap_results = {}
    for t in iou_thresholds:
        ap, _, _ = evaluate_map(all_predictions_per_threshold[t], total_gt, iou_threshold=t)
        ap_results[round(float(t), 2)] = ap

    map_50 = ap_results[0.5]
    map_75 = ap_results[0.75]
    map_50_95 = np.mean(list(ap_results.values()))

    # 速度
    avg_infer_time = np.mean(inference_times) if len(inference_times) > 0 else 0.0
    fps = 1.0 / avg_infer_time if avg_infer_time > 0 else 0.0

    print("\n" + "=" * 50)
    print("评测结果汇总")
    print("=" * 50)
    print(f"总帧数: {total_frames}")
    print(f"总 GT 数(目标存在帧): {total_gt}")
    print(f"TP@0.5: {tp_50}")
    print(f"FP@0.5: {fp_50}")
    print(f"FN@0.5: {fn_50}")
    print(f"TN@0.5: {tn_50}")
    print("-" * 50)
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Success Rate (IoU>0.5): {success_rate:.4f}")
    print(f"Precision@0.5: {precision_50:.4f}")
    print(f"Recall@0.5: {recall_50:.4f}")
    print(f"F1@0.5: {f1_50:.4f}")
    print("-" * 50)
    print(f"AP@0.50: {map_50:.4f}")
    print(f"AP@0.75: {map_75:.4f}")
    print(f"mAP@0.50: {map_50:.4f}")
    print(f"mAP@0.50:0.95: {map_50_95:.4f}")
    print("-" * 50)
    print(f"平均推理时间: {avg_infer_time * 1000:.2f} ms/frame")
    print(f"FPS: {fps:.2f}")
    print("=" * 50)

    print("\n各 IoU 阈值 AP:")
    for k, v in ap_results.items():
        print(f"AP@{k:.2f}: {v:.4f}")


if __name__ == "__main__":
    evaluate_antiuav_yolo_metrics(
        weights='/data1/code/xy/tph-yolov5/runs/train/tphv5l_antiuav_lsk2/weights/best.pt',
        data_dir='/data1/code/xy/datasets/Anti-UAV-video',
        imgsz=640,
        conf_thres=0.001,   # 算 mAP 时建议低一点
        iou_thres_nms=0.45
    )