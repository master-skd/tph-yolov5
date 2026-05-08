# video_predict.py
"""
YOLOv5 视频预测脚本 (修复版)
用法: python video_predict.py --weights best.pt --source input.mp4 --imgsz 1920

功能特点：
- 针对 1920x1080 视频，当 imgsz=1920 时，不做 Resize 缩放，仅做 Stride 对齐的 Padding。
- 支持实时显示 (--no-show 关闭)。
- 支持结果保存。
"""

import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
import sys
import os

# --- 环境设置: 添加项目根目录到 Python 路径 ---
FILE = Path(__file__).resolve()
ROOT = FILE.parent  # 项目根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# --- YOLOv5 模块导入 ---
# 请确保你的环境下有 models 和 utils 文件夹
try:
    from models.experimental import attempt_load
    from utils.general import (
        check_img_size, non_max_suppression,
        set_logging, colorstr, scale_coords
    )
    from utils.plots import plot_one_box
    from utils.torch_utils import select_device, time_sync
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保此脚本位于 YOLOv5 项目根目录下，并且包含 models 和 utils 文件夹。")
    sys.exit(1)


class VideoPredictor:
    def __init__(self, weights, imgsz=1920, conf_thres=0.25, iou_thres=0.45, device=''):
        """
        初始化视频预测器
        """
        set_logging()
        self.device = select_device(device)
        self.half = self.device.type != 'cpu'  # 半精度 FP16

        # 加载模型
        print(f"{colorstr('Loading model:')} {weights}")
        self.model = attempt_load(weights, map_location=self.device)
        self.stride = int(self.model.stride.max())

        # 检查图片尺寸 (确保是 stride 的倍数)
        self.imgsz = int(check_img_size(imgsz, s=self.stride))
        
        # 获取类别名
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        # FP16 设置
        if self.half:
            self.model.half()

        # 模型热身 (Warmup)
        if self.device.type != 'cpu':
            self.model(
                torch.zeros(1, 3, self.stride * 2, self.stride * 2)
                .to(self.device)
                .type_as(next(self.model.parameters()))
            )

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        print(f"{colorstr('Model info:')}")
        print(f"  Device: {self.device}")
        print(f"  Stride: {self.stride}")
        print(f"  Target Width (imgsz): {self.imgsz}")

    def preprocess(self, img0):
        """
        自定义预处理：
        1. 如果输入宽度 == target_w，则不进行 resize，只做 padding。
        2. 否则按比例缩放。
        3. Padding 确保尺寸是 Stride 的倍数。
        """
        h0, w0 = img0.shape[:2]
        img0_shape = (h0, w0)
        target_w = self.imgsz

        # 1. 缩放逻辑
        if w0 == target_w:
            # 完美匹配宽度，不缩放
            img = img0
            r = 1.0
        else:
            # 需要缩放
            r = target_w / w0
            new_w = target_w
            new_h = int(round(h0 * r))
            if new_w != w0 or new_h != h0:
                img = cv2.resize(img0, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                img = img0

        h, w = img.shape[:2]

        # 2. Padding 逻辑 (只补齐到 Stride 的倍数)
        pad_h = (self.stride - h % self.stride) % self.stride
        pad_w = (self.stride - w % self.stride) % self.stride

        dh = pad_h / 2
        dw = pad_w / 2

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        if pad_h > 0 or pad_w > 0:
            img = cv2.copyMakeBorder(
                img, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=(114, 114, 114)
            )

        # 3. 转换格式: HWC -> CHW, BGR -> RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        # 4. 转 Tensor 并归一化
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        ratio = (r, r)
        pad = (dw, dh)
        return img, img0_shape, ratio, pad

    def predict_frame(self, img):
        """单帧推理"""
        with torch.no_grad():
            pred = self.model(img, augment=False)[0]
            pred = non_max_suppression(
                pred,
                self.conf_thres,
                self.iou_thres,
                classes=None,
                agnostic=False
            )
        return pred[0] if pred[0] is not None else torch.empty(0, 6)

    def _scale_coords_back(self, detections, img0_shape, ratio, pad):
        """
        手动将坐标映射回原图 (对应 preprocess 的逻辑)
        """
        if detections is None or len(detections) == 0:
            return detections

        det = detections.clone()

        # 1. 去除 Padding
        det[:, [0, 2]] -= pad[0]  # x
        det[:, [1, 3]] -= pad[1]  # y

        # 2. 去除缩放
        det[:, :4] /= ratio[0]

        # 3. 边界截断 (Clip)
        h0, w0 = img0_shape
        det[:, 0].clamp_(0, w0)
        det[:, 1].clamp_(0, h0)
        det[:, 2].clamp_(0, w0)
        det[:, 3].clamp_(0, h0)

        return det

    def process_video(self, source, output_path=None, show_video=True, save_video=True):
        """处理视频主循环"""
        print(f"{colorstr('Opening video:')} {source}")
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {source}")

        # 获取视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"  Original: {width}x{height}, FPS: {fps_in:.2f}, Frames: {total_frames}")

        # --- 初始化 VideoWriter (修复点) ---
        out = None
        if save_video:
            # 确定输出路径
            src_path = Path(source)
            if output_path:
                out_p = Path(output_path)
                if out_p.suffix == '': # 如果看起来像目录
                    out_p.mkdir(parents=True, exist_ok=True)
                    final_out = out_p / f"{src_path.stem}_detect.mp4"
                else:
                    out_p.parent.mkdir(parents=True, exist_ok=True)
                    final_out = out_p
            else:
                final_out = src_path.parent / f"{src_path.stem}_detect.mp4"
            
            output_path = str(final_out)
            
            # 创建写入对象
            # 尝试使用 mp4v 编码
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))
            
            if not out.isOpened():
                print(f"{colorstr('red', 'Warning:')} 无法创建输出视频，尝试使用 avc1 编码...")
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                out = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))
            
            if out.isOpened():
                print(f"  Saving to: {output_path}")
            else:
                print(f"{colorstr('red', 'Error:')} 视频保存初始化失败！将只显示不保存。")
                save_video = False

        # --- 处理循环 ---
        frame_count = 0
        total_time = 0.0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % 10 == 0:
                    print(f"\rFrame: {frame_count}/{total_frames}", end="")

                t0 = time_sync()

                # 1. 预处理
                img, img0_shape, ratio, pad = self.preprocess(frame)

                # 2. 推理
                pred = self.predict_frame(img)

                # 3. 坐标还原
                det = self._scale_coords_back(pred, img0_shape, ratio, pad)

                # 4. 绘图
                if len(det):
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, frame, label=label, color=(0, 255, 0), line_thickness=2)

                t1 = time_sync()
                dt = t1 - t0
                total_time += dt
                curr_fps = 1.0 / dt if dt > 0 else 0

                # 显示信息
                cv2.putText(frame, f"FPS: {curr_fps:.1f}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # 5. 显示
                if show_video:
                    cv2.imshow('YOLOv5', frame)
                    if cv2.waitKey(1) == ord('q'):
                        print("\nInterrupted by user.")
                        break
                
                # 6. 保存
                if save_video and out:
                    out.write(frame)

        except KeyboardInterrupt:
            print("\nKeyboard Interrupt.")
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()

        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"\n{colorstr('Done.')} Average FPS: {avg_fps:.2f}")
        return avg_fps


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='weights path')
    parser.add_argument('--source', type=str, required=True, help='video source path')
    parser.add_argument('--output', type=str, default=None, help='output path')
    parser.add_argument('--imgsz', type=int, default=1920, help='inference size (width)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--no-show', action='store_true', help='do not show video')
    parser.add_argument('--no-save', action='store_true', help='do not save video')
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    predictor = VideoPredictor(
        weights=opt.weights,
        imgsz=opt.imgsz,
        conf_thres=opt.conf_thres,
        iou_thres=opt.iou_thres,
        device=opt.device
    )
    predictor.process_video(
        source=opt.source,
        output_path=opt.output,
        show_video=not opt.no_show,
        save_video=not opt.no_save
    )