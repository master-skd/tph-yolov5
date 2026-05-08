# video_predict_stage2.py
"""
YOLOv5 视频预测（FPV 稳定版 · Stage 2 · SMA Mode）

修改点：
- 移除指数衰减 (Exponential Decay)
- 采用多帧滑动平均 (Simple Moving Average)
- 漏检时填充 0.0 参与平均计算
"""

import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
import sys
from collections import deque
from tqdm import tqdm

# ---------------- 项目路径 ---------------- #
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, set_logging, colorstr
from utils.plots import plot_one_box
from utils.torch_utils import select_device


# ---------------- utils ---------------- #
def box_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)


def center_dist(box1, box2):
    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3]) / 2
    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5


# ---------------- Predictor ---------------- #
class VideoPredictor:
    def __init__(self, weights, imgsz, conf_thres, iou_thres,
                 conf_win, conf_avg_thres, iou_track, device):

        set_logging()
        self.device = select_device(device)
        self.half = False

        print(f"{colorstr('Loading model:')} {weights}")
        self.model = attempt_load(weights, map_location=self.device)
        self.stride = int(self.model.stride.max())
        if not self.half:
            self.model.float()
        self.imgsz = int(check_img_size(imgsz, s=self.stride))
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        if self.half:
            self.model.half()

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.stride * 20, self.stride * 20)
                       .to(self.device)
                       .type_as(next(self.model.parameters())))

        # ---------- parameters ---------- #
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.conf_win = conf_win           # 平均窗口大小 (N帧)
        self.conf_avg_thres = conf_avg_thres # 平均分阈值

        self.iou_track = iou_track
        self.max_center_dist = 120        
        self.max_tracks_per_class = 2     

        # self.decay = 0.8  <-- 已移除
        self.max_miss = conf_win * 2       # 允许最大连续丢失帧数
        self.warmup = conf_win             # 预热期

        self.tracks = []

        print(colorstr('Tracking params (SMA Mode):'))
        print(f'  Win size: {self.conf_win}')
        print(f'  Avg Thres: {self.conf_avg_thres}')
        print(f'  IoU track: {self.iou_track}')

    # ---------- preprocess ---------- #
    def preprocess(self, img0):
        h0, w0 = img0.shape[:2]
        target_w = self.imgsz
        r = 1.0
        img = img0
        h, w = img.shape[:2]
        pad_h = (self.stride - h % self.stride) % self.stride
        pad_w = (self.stride - w % self.stride) % self.stride
        dh, dw = pad_h / 2, pad_w / 2

        img = cv2.copyMakeBorder(
            img,
            int(round(dh - 0.1)), int(round(dh + 0.1)),
            int(round(dw - 0.1)), int(round(dw + 0.1)),
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = torch.from_numpy(np.ascontiguousarray(img)).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0

        return img.unsqueeze(0), (h0, w0), (r, r), (dw, dh)

    # ---------- predict ---------- #
    def predict_frame(self, img):
        with torch.no_grad():
            pred = self.model(img)[0]
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        return pred[0] if pred[0] is not None else torch.empty(0, 6)

    # ---------- scale back ---------- #
    @staticmethod
    def scale_back(det, shape, ratio, pad):
        if len(det) == 0:
            return det

        det = det.clone()  # ✅ 防止 inplace 影响原 tensor

        det[:, [0, 2]] -= pad[0]
        det[:, [1, 3]] -= pad[1]
        det[:, :4] /= ratio[0]

        h, w = shape
        det[:, 0].clamp_(0, w)
        det[:, 2].clamp_(0, w)
        det[:, 1].clamp_(0, h)
        det[:, 3].clamp_(0, h)

        det[:, :4] = det[:, :4].round()  # ✅ 只 round bbox
        return det

    # ---------- main loop ---------- #
    def process_video(self, source, output, show, save):
        cap = cv2.VideoCapture(str(source))
        w, h = int(cap.get(3)), int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        # ---------- FPS 兜底 ----------
        if fps is None or fps <= 0 or fps > 60:
            print(colorstr('Warning:'), f'Invalid FPS ({fps}), fallback to 25')
            fps = 25
        else:
            fps = int(round(fps))

        out = None
        if save:
            src = Path(source)

            # ✅ 1️⃣ 决定输出目录
            if output is None:
                # 没传 --output，用 source 同级 detected
                output_dir = src.parent / 'detected'
            else:
                # 传了 --output，一律当作目录
                output_dir = Path(output)

            output_dir.mkdir(parents=True, exist_ok=True)

            # ✅ 2️⃣ 自动生成输出文件名
            output_path = output_dir / f"{src.stem}_detect.mp4"

            # ✅ 3️⃣ 打开 VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
            if not out.isOpened():
                raise RuntimeError(f'Failed to open VideoWriter: {output_path}')

            print(colorstr('Output video:'), output_path)

        pbar = tqdm(desc='Processing', unit='frame')
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            img, shape, ratio, pad = self.preprocess(frame)
            det = self.scale_back(self.predict_frame(img), shape, ratio, pad)

            # 1️⃣ 初始化更新状态
            # 在每一帧开始时，先将所有 track 标记为 "未更新"
            for t in self.tracks:
                t['updated'] = False

            # 2️⃣ 匹配与更新 (Match & Update)
            for *xyxy, conf, cls in det:
                box = list(map(float, xyxy))
                cls = int(cls)
                conf = float(conf)

                # print(f"Debug - Frame {frame_idx}: Raw Conf = {conf}")

                best_t, best_score = None, 0.0
                for t in self.tracks:
                    if t['cls'] != cls:
                        continue
                    iou = box_iou(box, t['box'])
                    dist = center_dist(box, t['box'])

                    if iou > self.iou_track or dist < self.max_center_dist:
                        score = iou - dist * 1e-4
                        if score > best_score:
                            best_score = score
                            best_t = t

                if best_t is not None:
                    # ✅ 匹配成功
                    best_t['box'] = box
                    best_t['hist'].append(conf) # 存入当前真实置信度
                    best_t['miss'] = 0
                    best_t['updated'] = True    # 标记为已更新
                else:
                    # ✅ 新目标
                    same_cls = [t for t in self.tracks if t['cls'] == cls]
                    new_track = {
                        'box': box,
                        'cls': cls,
                        'hist': deque([conf], maxlen=self.conf_win),
                        'miss': 0,
                        'updated': True, # 新目标也算作本帧已更新,
                        'age': 1
                    }
                    if len(same_cls) < self.max_tracks_per_class:
                        self.tracks.append(new_track)
                    else:
                        # 如果满员，替换掉平均分最低的那个
                        weakest = min(same_cls, key=lambda t: sum(t['hist'])/len(t['hist']))
                        if conf > (sum(weakest['hist'])/len(weakest['hist'])):
                            self.tracks.remove(weakest)
                            self.tracks.append(new_track)

            # 3️⃣ 处理未匹配的 Tracks (漏检处理)
            for t in self.tracks:
                if not t['updated']:
                    t['miss'] += 1
                    # --- 这里定义“漏检时的合理值” ---
                    # 选项 A: 0.0 (表示完全没看到，平均分会自然下降) -> 推荐，反应灵敏
                    # 选项 B: t['hist'][-1] (沿用上一帧分数，平均分保持不变，直到 max_miss 删除) -> 适合需要强行死磕的情况
                    fill_value = 0.0 
                    t['hist'].append(fill_value)

            # 4️⃣ 绘制 & 清理
            alive = []
            for t in self.tracks:
                if t['miss'] > self.max_miss:
                    continue
                
                # 计算简单滑动平均 (Simple Moving Average)
                avg_conf = sum(t['hist']) / len(t['hist'])

                # 只有在 warm-up 期间或者 平均置信度 达标时才显示
                if frame_idx <= self.warmup or avg_conf >= self.conf_avg_thres:
                    
                    if t['miss'] == 0:
                        # ✅ 情况A：真实检测到 (Green)
                        # 显示：真实置信度 (Raw)
                        current_raw = t['hist'][-1]
                        label = f"{self.names[t['cls']]} {current_raw:.2f}"
                        
                    else:
                        # ✅ 情况B：漏检补帧 (Red)
                        # 显示：平均置信度 (Avg) -> 这样你知道它是靠"平均分"活下来的
                        label = f"{self.names[t['cls']]} {avg_conf:.2f}"
                    box_color = (0, 255, 0)
                    plot_one_box(t['box'], frame, label=label,
                                 color=box_color, line_thickness=2)
                
                alive.append(t)
            self.tracks = alive

            if show:
                cv2.imshow('YOLOv5 SMA', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if save:
                out.write(frame)

            pbar.update(1)

        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        pbar.close()


# ---------------- args ---------------- #
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--imgsz', type=int, default=1920)
    parser.add_argument('--conf-thres', type=float, default=0.4)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    
    # 窗口大小建议：FPS高可以设大一点(10)，FPS低设小一点(3-5)
    parser.add_argument('--conf-win', type=int, default=5, help='Moving Average Window Size')
    # 平均分阈值：如果5帧里有一帧没检测到(0分)，平均分会被拉低，所以这个值通常比 conf-thres 低一点
    parser.add_argument('--conf-avg-thres', type=float, default=0.25)
    
    parser.add_argument('--iou-track', type=float, default=0.2)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    return parser.parse_args()


def main(opt):
    predictor = VideoPredictor(
        opt.weights, opt.imgsz,
        opt.conf_thres, opt.iou_thres,
        opt.conf_win, opt.conf_avg_thres,
        opt.iou_track, opt.device
    )
    predictor.process_video(
        opt.source, opt.output,
        show=not opt.no_show,
        save=not opt.no_save
    )


if __name__ == '__main__':
    main(parse_opt())

'''
python predict.py \
  --weights /data1/code/xy/v5world/weights/best.pt \
  --source /data1/code/xy/datasets/wurenyuan/predict.mp4 \
  --imgsz 1280 \
  --conf-thres 0.25 \
  --iou-thres 0.25 \
  --conf-win 5 \
  --conf-avg-thres 0.25 \
  --iou-track 0.5 \
  --no-show \
  --device 0
'''