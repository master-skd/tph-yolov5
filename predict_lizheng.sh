BASE=/data1/code/xy/datasets/lizheng_nolabel

find "$BASE" -name "*.mp4" ! -name "*_detect.mp4" | while read v; do
  echo "Processing $v"
  python predict.py \
  --weights /data1/code/xy/tph-yolov5/runs/train/tphv5l_anti-lizheng2/weights/best.pt \
  --source "$v" \
  --imgsz 1920 \
  --conf-thres 0.2 \
  --iou-thres 0.45 \
  --conf-win 5 \
  --conf-avg-thres 0.45 \
  --iou-track 0.1 \
  --no-show \
  --device 0
done


'''
find "$BASE" -name "*.mp4" ! -name "*_detect.mp4" | while read v; do
  echo "Processing $v"
  python predict_old.py \
  --weights /data1/code/xy/tph-yolov5/runs/train/tphv5l_lizheng_merge2/weights/best.pt \
  --source "$v" \
  --no-show \
  --device 3 \
  --output /data1/code/xy/datasets/lizheng_nolabel/detected
done
'''