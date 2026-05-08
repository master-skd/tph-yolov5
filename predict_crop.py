import argparse
import cv2
import torch
import torchvision
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

# ---------------- 项目路径 ---------------- #
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, set_logging, colorstr, scale_coords, clip_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device


class VideoPredictorCrop:
    def __init__(self, weights, imgsz, conf_thres, iou_thres, device, p, q, k, crop_size=640):
        set_logging()
        self.device = select_device(device)
        self.half = self.device.type != 'cpu'  # half precision only on CUDA

        print(f"{colorstr('Loading model:')} {weights}")
        self.model = attempt_load(weights, map_location=self.device)
        self.stride = int(self.model.stride.max())
        if self.half:
            self.model.half()
        else:
            self.model.float()
            
        self.imgsz = int(check_img_size(imgsz, s=self.stride))
        self.crop_imgsz = int(check_img_size(crop_size, s=self.stride))
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        # ---------- parameters ---------- #
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.p = p
        self.q = q
        self.k = k
        self.crop_size = crop_size

        # Warmup
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.stride * 20, self.stride * 20)
                       .to(self.device)
                       .type_as(next(self.model.parameters())))

    def preprocess(self, img0, imgsz):
        """
        Standard YOLOv5 preprocessing: resize with padding
        """
        img = self.letterbox(img0, imgsz, stride=self.stride)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        return img.unsqueeze(0)

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)

    def predict_frame(self, img_tensor, conf_thres):
        with torch.no_grad():
            pred = self.model(img_tensor)[0]
            pred = non_max_suppression(pred, conf_thres, self.iou_thres)
        return pred[0] if pred[0] is not None else torch.empty(0, 6, device=img_tensor.device)

    def get_crop(self, img0, box):
        h, w = img0.shape[:2]
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        half_size = self.crop_size / 2
        
        left = max(0, cx - half_size)
        top = max(0, cy - half_size)
        right = min(w, left + self.crop_size)
        bottom = min(h, top + self.crop_size)
        
        # Adjust if hitting right/bottom edges to maintain crop_size if possible
        if right == w:
            left = max(0, w - self.crop_size)
        if bottom == h:
            top = max(0, h - self.crop_size)
            
        left, top, right, bottom = map(int, [left, top, right, bottom])
        crop = img0[top:bottom, left:right]
        return crop, (left, top, right, bottom)

    def is_box_center_in_regions(self, boxes, regions):
        if len(boxes) == 0 or not regions:
            return torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)

        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        mask = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)
        for left, top, right, bottom in regions:
            mask |= (cx >= left) & (cx <= right) & (cy >= top) & (cy <= bottom)
        return mask

    def apply_final_nms(self, det):
        if len(det) <= 1:
            return det

        max_wh = 10000
        boxes = det[:, :4] + det[:, 5:6] * max_wh
        keep = torchvision.ops.nms(boxes.float(), det[:, 4].float(), self.iou_thres)
        return det[keep]

    def infer_image(self, img0, return_meta=False):
        img_tensor = self.preprocess(img0, self.imgsz)
        det = self.predict_frame(img_tensor, self.conf_thres)

        final_det = torch.empty(0, 6, device=self.device)
        crop_regions = []
        crop_dets = []
        crop_triggered = False
        second_pass_boxes = 0
        first_pass_boxes = int(len(det))

        if len(det) > 0:
            det = det.clone()
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img0.shape).round()

            if len(det) >= self.k:
                for d in det:
                    xyxy = d[:4]
                    conf = float(d[4].item())
                    if conf < self.p:
                        crop_triggered = True
                        crop, crop_region = self.get_crop(img0, xyxy.tolist())
                        offset_x, offset_y, _, _ = crop_region
                        crop_regions.append(crop_region)

                        crop_tensor = self.preprocess(crop, self.crop_imgsz)
                        crop_det = self.predict_frame(crop_tensor, self.q)

                        if len(crop_det) > 0:
                            crop_det = crop_det.clone()
                            crop_det[:, :4] = scale_coords(crop_tensor.shape[2:], crop_det[:, :4], crop.shape).round()
                            crop_det[:, [0, 2]] += offset_x
                            crop_det[:, [1, 3]] += offset_y
                            clip_coords(crop_det[:, :4], img0.shape)
                            second_pass_boxes += int(len(crop_det))
                            crop_dets.append(crop_det)

                if crop_regions:
                    outside_mask = ~self.is_box_center_in_regions(det[:, :4], crop_regions)
                    final_parts = [det[outside_mask]]
                    if crop_dets:
                        final_parts.append(torch.cat(crop_dets, dim=0))
                    final_det = torch.cat(final_parts, dim=0)
                    final_det = self.apply_final_nms(final_det)
                else:
                    final_det = det
            else:
                final_det = det

        if return_meta:
            meta = {
                'first_pass_boxes': first_pass_boxes,
                'crop_triggered': crop_triggered,
                'crop_windows': len(crop_regions),
                'second_pass_boxes': second_pass_boxes,
                'final_boxes': int(len(final_det)),
            }
            return final_det, meta

        return final_det

    def process_video(self, source, output, show, save):
        cap = cv2.VideoCapture(str(source))
        w, h = int(cap.get(3)), int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0 or fps > 60:
            fps = 25
        else:
            fps = int(round(fps))

        out = None
        if save:
            src = Path(source)
            output_dir = Path(output) if output else src.parent / 'detected_crop'
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{src.stem}_crop.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
            print(colorstr('Output video:'), output_path)

        pbar = tqdm(desc='Processing', unit='frame')
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img0 = frame.copy()
            final_det = self.infer_image(img0)

            # 3. Draw final detections
            for det_box in final_det:
                *xyxy, conf, cls = det_box
                label = f"{self.names[int(cls)]} {conf:.2f}"
                plot_one_box(xyxy, frame, label=label, color=(0, 255, 0), line_thickness=2)

            if show:
                cv2.imshow('YOLOv5 Crop Inference', frame)
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


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--imgsz', type=int, default=1920)
    parser.add_argument('--conf-thres', type=float, default=0.15, help='Initial confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    
    # New logic parameters
    parser.add_argument('--p', type=float, default=0.5, help='Confidence threshold to trigger crop')
    parser.add_argument('--q', type=float, default=0.45, help='Confidence threshold for second-pass crop inference')
    parser.add_argument('--k', type=int, default=3, help='Minimum number of boxes to trigger crop logic')
    parser.add_argument('--crop-size', type=int, default=640, help='Size of the cropped region')
    
    return parser.parse_args()


def main(opt):
    predictor = VideoPredictorCrop(
        opt.weights, opt.imgsz,
        opt.conf_thres, opt.iou_thres,
        opt.device, opt.p, opt.q, opt.k, opt.crop_size
    )
    predictor.process_video(
        opt.source, opt.output,
        show=not opt.no_show,
        save=not opt.no_save
    )


if __name__ == '__main__':
    main(parse_opt())
