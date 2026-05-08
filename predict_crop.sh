#!/bin/bash

# 基础目录
BASE=/data1/code/xy/datasets/wurenyuan/测试视频

# 查找视频并进行推理
find "$BASE" -name "20260420.mp4" | while read v; do
    echo "Processing $v with Crop Logic"
    python predict_crop.py \
      --weights /data1/code/xy/tph-yolov5/runs/train/tphv5l_antiuav_cop_lsk_api/weights/best.pt \
      --source "$v" \
      --imgsz 1920 \
      --conf-thres 0.45 \
      --iou-thres 0.45 \
      --p 0.5 \
      --q 0.45 \
      --k 2 \
      --crop-size 640 \
      --no-show \
      --device 3 \
      --output /data1/code/xy/datasets/wurenyuan/测试视频/detected_crop
done
