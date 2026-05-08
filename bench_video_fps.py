import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression
from utils.torch_utils import select_device, time_sync


def parse_opt():
    parser = argparse.ArgumentParser(description='Benchmark YOLOv5 PT video FPS on a single GPU')
    parser.add_argument(
        '--weights',
        type=str,
        default='/data1/code/xy/tph-yolov5/runs/train/tphv5l_antiuav_cop_lsk_api/weights/best.pt',
        help='path to .pt weights',
    )
    parser.add_argument('--source', type=str, default='', help='video path, rtsp url, or camera id like 0')
    parser.add_argument('--image', type=str, default='', help='single image path to repeat for benchmarking')
    parser.add_argument('--synthetic', action='store_true', help='repeat a synthetic frame for benchmarking')
    parser.add_argument('--frame-size', nargs=2, type=int, default=[1080, 1920], help='source frame size h w')
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda device, e.g. cuda:0 or 0 or cpu')
    parser.add_argument('--imgsz', nargs=2, type=int, default=[1080, 1920], help='inference size h w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='maximum detections per frame')
    parser.add_argument('--warmup', type=int, default=30, help='number of warmup iterations')
    parser.add_argument('--max-frames', type=int, default=300, help='max frames to benchmark, <=0 means all')
    parser.add_argument('--log-interval', type=int, default=50, help='print running FPS every N frames')
    parser.add_argument('--augment', action='store_true', help='use augmented inference')
    parser.add_argument('--fp32', action='store_true', help='force FP32 inference on CUDA')
    return parser.parse_args()


def open_source(source):
    source = str(source)
    if source.isdigit():
        return cv2.VideoCapture(int(source))
    return cv2.VideoCapture(source)


def prepare_static_frame(image_path, frame_size):
    frame_h, frame_w = frame_size
    if image_path:
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f'Failed to read image: {image_path}')
        frame = cv2.resize(frame, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
        return frame, 'image'

    # Use a stable synthetic frame instead of random noise, so NMS cost is not overly distorted.
    frame = np.full((frame_h, frame_w, 3), 114, dtype=np.uint8)
    cv2.putText(
        frame,
        'Synthetic 1920x1080 Benchmark Frame',
        (40, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (200, 200, 200),
        3,
        cv2.LINE_AA,
    )
    cv2.rectangle(frame, (160, 120), (540, 420), (90, 130, 200), thickness=-1)
    cv2.circle(frame, (1200, 560), 180, (170, 170, 170), thickness=-1)
    cv2.line(frame, (0, frame_h - 120), (frame_w, frame_h - 120), (150, 150, 150), thickness=2)
    return frame, 'synthetic'


def preprocess_frame(frame, imgsz, stride, device, half):
    # Use fixed letterbox shape so the benchmark matches a fixed deployment input.
    img = letterbox(frame, new_shape=imgsz, stride=stride, auto=False)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    img = img.unsqueeze(0)
    return img


@torch.no_grad()
def run(opt):
    device = select_device(opt.device)
    half = device.type != 'cpu' and not opt.fp32

    print(f'Loading weights: {opt.weights}')
    model = attempt_load(opt.weights, map_location=device)
    model.eval()
    if half:
        model.half()
    else:
        model.float()

    stride = int(model.stride.max())
    imgsz = check_img_size(opt.imgsz, s=stride)
    imgsz = tuple(imgsz)
    frame_size = tuple(opt.frame_size)

    cap = None
    static_frame = None
    if opt.source:
        cap = open_source(opt.source)
        if not cap.isOpened():
            raise FileNotFoundError(f'Failed to open source: {opt.source}')
        source_mode = 'video'
        source_name = opt.source
        source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        if opt.max_frames <= 0:
            opt.max_frames = 300
        static_frame, source_mode = prepare_static_frame(opt.image, frame_size)
        source_name = opt.image if opt.image else 'synthetic-frame'
        source_h, source_w = static_frame.shape[:2]
        source_fps = 0.0
        total_frames = opt.max_frames

    print('-' * 60)
    print(f'Source       : {source_name}')
    print(f'Source mode  : {source_mode}')
    print(f'Video shape  : {source_w}x{source_h}')
    print(f'Video FPS    : {source_fps:.3f}' if source_fps > 0 else 'Video FPS    : unknown')
    print(f'Total frames : {total_frames}' if total_frames > 0 else 'Total frames : unknown')
    print(f'Device       : {device}')
    print(f'Precision    : {"FP16" if half else "FP32"}')
    print(f'Model stride : {stride}')
    print(f'Input shape  : {imgsz[1]}x{imgsz[0]} (w x h)')
    print('-' * 60)

    if device.type != 'cpu':
        cudnn.benchmark = True
        warmup_input = torch.zeros(1, 3, imgsz[0], imgsz[1], device=device)
        warmup_input = warmup_input.half() if half else warmup_input.float()
        for _ in range(max(opt.warmup, 0)):
            _ = model(warmup_input, augment=opt.augment)[0]

    frames = 0
    total_det = 0
    t_read = 0.0
    t_pre = 0.0
    t_inf = 0.0
    t_nms = 0.0
    t_post = 0.0
    t_total = 0.0

    print('Start benchmark...')

    while True:
        if opt.max_frames > 0 and frames >= opt.max_frames:
            break

        t0 = time_sync()
        t1 = time_sync()
        if source_mode == 'video':
            ok, frame = cap.read()
        else:
            ok, frame = True, static_frame.copy()
        t2 = time_sync()
        if not ok:
            break
        t_read += t2 - t1

        t3 = time_sync()
        img = preprocess_frame(frame, imgsz, stride, device, half)
        t4 = time_sync()
        t_pre += t4 - t3

        pred = model(img, augment=opt.augment)[0]
        t5 = time_sync()
        t_inf += t5 - t4

        pred = non_max_suppression(
            pred,
            conf_thres=opt.conf_thres,
            iou_thres=opt.iou_thres,
            max_det=opt.max_det,
        )
        t6 = time_sync()
        t_nms += t6 - t5

        det_count = sum(len(det) for det in pred if det is not None)
        total_det += det_count
        t_post += time_sync() - t6

        frames += 1
        t_total += time_sync() - t0

        if opt.log_interval > 0 and frames % opt.log_interval == 0:
            running_fps = frames / max(t_total, 1e-9)
            print(f'Processed {frames} frames, running end-to-end FPS: {running_fps:.2f}')

    if cap is not None:
        cap.release()

    if frames == 0:
        raise RuntimeError('No frames were processed. Please check the video path or camera source.')

    avg_read_ms = t_read * 1000 / frames
    avg_pre_ms = t_pre * 1000 / frames
    avg_inf_ms = t_inf * 1000 / frames
    avg_nms_ms = t_nms * 1000 / frames
    avg_post_ms = t_post * 1000 / frames
    avg_total_ms = t_total * 1000 / frames

    overall_fps = frames / max(t_total, 1e-9)
    pipeline_fps = frames / max(t_pre + t_inf + t_nms + t_post, 1e-9)
    model_fps = frames / max(t_inf + t_nms, 1e-9)
    forward_fps = frames / max(t_inf, 1e-9)

    print('\nBenchmark done')
    print('-' * 60)
    print(f'Frames benchmarked : {frames}')
    print(f'Avg detections/frame: {total_det / frames:.3f}')
    print(f'Avg read time      : {avg_read_ms:.3f} ms')
    print(f'Avg preprocess     : {avg_pre_ms:.3f} ms')
    print(f'Avg inference      : {avg_inf_ms:.3f} ms')
    print(f'Avg NMS            : {avg_nms_ms:.3f} ms')
    print(f'Avg postprocess    : {avg_post_ms:.3f} ms')
    print(f'Avg total/frame    : {avg_total_ms:.3f} ms')
    print('-' * 60)
    print(f'End-to-end FPS     : {overall_fps:.3f}')
    print(f'Pipeline FPS       : {pipeline_fps:.3f}  (preprocess + inference + NMS + post)')
    print(f'Model FPS          : {model_fps:.3f}  (inference + NMS)')
    print(f'Forward-only FPS   : {forward_fps:.3f}  (inference only)')
    if source_mode != 'video':
        print('Note             : static-frame mode excludes real video decode cost')
    print('-' * 60)


if __name__ == '__main__':
    run(parse_opt())
