import argparse
import json
import os
import sys
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from predict_crop import VideoPredictorCrop
from utils.datasets import create_dataloader
from utils.general import (LOGGER, check_dataset, check_requirements, check_suffix, check_yaml, coco80_to_coco91_class,
                           colorstr, increment_path, non_max_suppression, print_args, scale_coords, xywh2xyxy)
from utils.metrics import ConfusionMatrix, ap_per_class
from utils.plots import plot_images
from utils.torch_utils import time_sync
from val import process_batch, save_one_json, save_one_txt


def preprocess_crop_like_val(predictor, crop):
    """Match val.py-style letterbox settings for crop refinement."""
    img = predictor.letterbox(crop, predictor.crop_imgsz, auto=False, scaleup=False, stride=predictor.stride)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(predictor.device)
    img = img.half() if predictor.half else img.float()
    img /= 255.0
    return img.unsqueeze(0)


def refine_with_crop(predictor, det, img0, single_cls=False):
    """Refine native-space detections with a second crop pass."""
    final_det = det
    crop_regions = []
    crop_dets = []
    crop_triggered = False
    second_pass_boxes = 0
    first_pass_boxes = int(len(det))

    if len(det) >= predictor.k:
        for d in det:
            conf = float(d[4].item())
            if conf < predictor.p:
                crop_triggered = True
                crop, crop_region = predictor.get_crop(img0, d[:4].tolist())
                offset_x, offset_y, _, _ = crop_region
                crop_regions.append(crop_region)

                crop_tensor = preprocess_crop_like_val(predictor, crop)
                with torch.no_grad():
                    crop_out = predictor.model(crop_tensor)[0]
                crop_out = non_max_suppression(
                    crop_out,
                    predictor.q,
                    predictor.iou_thres,
                    multi_label=True,
                    agnostic=single_cls,
                )
                crop_det = crop_out[0] if crop_out[0] is not None else torch.empty(0, 6, device=predictor.device)

                if len(crop_det) > 0:
                    crop_det = crop_det.clone()
                    if single_cls:
                        crop_det[:, 5] = 0
                    scale_coords(crop_tensor.shape[2:], crop_det[:, :4], crop.shape)
                    crop_det[:, [0, 2]] += offset_x
                    crop_det[:, [1, 3]] += offset_y
                    second_pass_boxes += int(len(crop_det))
                    crop_dets.append(crop_det)

        if crop_regions:
            outside_mask = ~predictor.is_box_center_in_regions(det[:, :4], crop_regions)
            final_parts = [det[outside_mask]]
            if crop_dets:
                final_parts.append(torch.cat(crop_dets, dim=0))
            final_det = torch.cat(final_parts, dim=0)
            final_det = predictor.apply_final_nms(final_det)

    meta = {
        'first_pass_boxes': first_pass_boxes,
        'crop_triggered': crop_triggered,
        'crop_windows': len(crop_regions),
        'second_pass_boxes': second_pass_boxes,
        'final_boxes': int(len(final_det)),
    }
    return final_det, meta


@torch.no_grad()
def run(data,
        weights,
        batch_size=1,
        imgsz=1920,
        conf_thres=0.45,
        iou_thres=0.45,
        task='val',
        device='',
        single_cls=False,
        verbose=False,
        save_txt=False,
        save_conf=False,
        save_json=False,
        project=ROOT / 'runs/val-crop',
        name='exp',
        exist_ok=False,
        half=False,
        plots=False,
        p=0.5,
        q=0.45,
        k=2,
        crop_size=640):
    check_suffix(weights, '.pt')
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    data = check_dataset(data)
    task = task if task in ('train', 'val', 'test') else 'val'
    is_coco = isinstance(data.get(task), str) and (
        'coco/val2017.txt' in data[task] or 'coco.yaml' in str(data)
    )

    img_id_map = {}
    if save_json and not is_coco:
        anno_name = f'{task}.json'
        anno_path = Path(data['path']) / 'annotations' / anno_name
        if anno_path.exists():
            with open(anno_path, 'r') as f:
                anno = json.load(f)
            for img_info in anno['images']:
                img_id_map[Path(img_info['file_name']).name] = img_info['id']
            LOGGER.info(f'Loaded {len(img_id_map)} image ID mappings from {anno_path}')
        else:
            LOGGER.warning(f'Annotation file {anno_path} not found. image_id may be incorrect.')

    predictor = VideoPredictorCrop(weights, imgsz, conf_thres, iou_thres, device, p, q, k, crop_size)
    model = predictor.model
    device = predictor.device
    gs = max(int(model.stride.max()), 32)
    imgsz = predictor.imgsz
    half &= device.type != 'cpu'
    predictor.half = half
    model.half() if half else model.float()

    dataloader = create_dataloader(data[task], imgsz, batch_size, gs, single_cls, pad=0.5, rect=True,
                                   prefix=colorstr(f'{task}: '))[0]

    model.eval()
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    nc = 1 if single_cls else int(data['nc'])
    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    niou = iouv.numel()

    confusion_matrix = ConfusionMatrix(nc=nc)
    names = model.names if hasattr(model, 'names') else model.module.names
    names = names if isinstance(names, dict) else {i: n for i, n in enumerate(names)}
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')

    seen, dt = 0, 0.0
    jdict, stats, ap, ap_class = [], [], [], []
    crop_stats = {
        'images': 0,
        'triggered_images': 0,
        'crop_windows': 0,
        'first_pass_boxes': 0,
        'second_pass_boxes': 0,
        'final_boxes': 0,
    }

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        t1 = time_sync()
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()
        img /= 255.0
        targets = targets.to(device)
        _, _, height, width = img.shape
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)

        out, _ = model(img, augment=False)
        out = non_max_suppression(out, conf_thres, iou_thres, multi_label=True, agnostic=single_cls)

        for si, (path_str, pred) in enumerate(zip(paths, out)):
            path = Path(path_str)
            shape = shapes[si][0]
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []
            seen += 1

            if single_cls and len(pred):
                pred[:, 5] = 0

            predn = pred.clone()
            if len(predn):
                scale_coords(img[si].shape[1:], predn[:, :4], shape, shapes[si][1])

            meta = {
                'first_pass_boxes': int(len(predn)),
                'crop_triggered': False,
                'crop_windows': 0,
                'second_pass_boxes': 0,
                'final_boxes': int(len(predn)),
            }

            should_refine = len(predn) >= predictor.k and bool((predn[:, 4] < predictor.p).any())
            if should_refine:
                img0 = cv2.imread(str(path))
                if img0 is None:
                    LOGGER.warning(f'Failed to read image for crop refinement: {path}')
                else:
                    predn, meta = refine_with_crop(predictor, predn, img0, single_cls=single_cls)
                    if single_cls and len(predn):
                        predn[:, 5] = 0

            crop_stats['images'] += 1
            crop_stats['triggered_images'] += int(meta['crop_triggered'])
            crop_stats['crop_windows'] += meta['crop_windows']
            crop_stats['first_pass_boxes'] += meta['first_pass_boxes']
            crop_stats['second_pass_boxes'] += meta['second_pass_boxes']
            crop_stats['final_boxes'] += meta['final_boxes']

            if len(predn) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            if single_cls:
                predn[:, 5] = 0

            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shape, shapes[si][1])
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            else:
                correct = torch.zeros(predn.shape[0], niou, dtype=torch.bool)

            stats.append((correct.cpu(), predn[:, 4].cpu(), predn[:, 5].cpu(), tcls))

            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')
            if save_json:
                save_one_json(predn, jdict, path, class_map, img_id_map=img_id_map if not is_coco else None)

        dt += time_sync() - t1

        if plots and batch_i < 3:
            label_path = save_dir / f'val_batch{batch_i}_labels.jpg'
            Thread(target=plot_images, args=(img, targets, paths, label_path, names), daemon=True).start()

    stats = [np.concatenate(x, 0) for x in zip(*stats)] if stats else []
    mp = mr = map50 = map95 = 0.0
    p_vals, r_vals, ap50 = np.array([]), np.array([]), np.array([])
    if len(stats) and stats[0].any():
        p_vals, r_vals, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)
        mp, mr, map50, map95 = p_vals.mean(), r_vals.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)
    else:
        nt = torch.zeros(1)

    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4
    LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map95))

    if (verbose or nc < 50) and nc > 1 and len(stats) and len(ap_class):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p_vals[i], r_vals[i], ap50[i], ap[i]))

    avg_infer_ms = dt / max(seen, 1) * 1e3
    trigger_ratio = crop_stats['triggered_images'] / max(crop_stats['images'], 1)
    LOGGER.info(
        'Crop summary: triggered %.1f%% images, %d crop windows, %d first-pass boxes, %d second-pass boxes, %d final boxes',
        trigger_ratio * 100.0,
        crop_stats['crop_windows'],
        crop_stats['first_pass_boxes'],
        crop_stats['second_pass_boxes'],
        crop_stats['final_boxes'],
    )
    LOGGER.info(f'Speed: {avg_infer_ms:.1f}ms crop inference per image at shape (1, 3, {imgsz}, {imgsz})')

    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))

    if save_json and len(jdict):
        pred_json = str(save_dir / f'{Path(weights).stem}_predictions.json')
        anno_json = str(Path(data['path']) / 'annotations' / f'{task}.json')
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:
            check_requirements(['pycocotools'])
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)
            pred = anno.loadRes(pred_json)
            evaluator = COCOeval(anno, pred, 'bbox')
            if is_coco:
                evaluator.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
            map95, map50 = evaluator.stats[:2]
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")
    maps = np.zeros(nc) + map95
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map95), maps, avg_infer_ms, crop_stats


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', type=str, required=True, help='model.pt path')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size for labels/dataloader')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=1920, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='initial confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val or test')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--plots', action='store_true', help='save plots')
    parser.add_argument('--project', default=ROOT / 'runs/val-crop', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--p', type=float, default=0.5, help='confidence threshold to trigger crop')
    parser.add_argument('--q', type=float, default=0.45, help='confidence threshold for second-pass crop inference')
    parser.add_argument('--k', type=int, default=2, help='minimum number of boxes to trigger crop logic')
    parser.add_argument('--crop-size', type=int, default=640, help='size of the cropped region')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
