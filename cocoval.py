import argparse
import json
import os
import time
from pathlib import Path
from typing import Tuple, List
 
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
 
 
def validate_json_file(file_path: Path) -> Tuple[bool, str]:
    """验证JSON文件：存在性、文件类型、JSON格式合法性"""
    if not file_path.exists():
        return False, f"文件不存在 → {file_path}"
    if not file_path.is_file():
        return False, f"不是有效文件 → {file_path}"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json.load(f)
        return True, "JSON文件验证通过"
    except json.JSONDecodeError as e:
        return False, f"JSON格式错误 → {str(e)}（文件：{file_path}）"
    except Exception as e:
        return False, f"文件读取失败 → {str(e)}（文件：{file_path}）"
 
 
def print_iou50_size_metrics(stats: np.ndarray) -> None:
    """
    单独打印 IoU=0.5 阈值下大、中、小目标的 AP 和 AR 指标
    COCOeval.stats 数组索引对应关系（关键）：
    - 1: AP@0.5（整体）
    - 3: AP@0.5（小目标，面积 < 32²）
    - 4: AP@0.5（中目标，32² ≤ 面积 < 96²）
    - 5: AP@0.5（大目标，面积 ≥ 96²）
    - 7: AR@0.5（整体）
    - 9: AR@0.5（小目标）
    - 10: AR@0.5（中目标）
    - 11: AR@0.5（大目标）
    """
    # 定义尺寸分类的中文说明
    size_desc = {
        "small": "小目标（面积 < 32×32）",
        "medium": "中目标（32×32 ≤ 面积 < 96×96）",
        "large": "大目标（面积 ≥ 96×96）"
    }
 
    print("\n" + "=" * 60)
    print("📊 IoU=0.5 阈值下 大/中/小目标 专项评估结果")
    print("=" * 60)
    print(f"{'目标尺寸':<20} {'AP@0.5':<10} {'AR@0.5':<10}")
    print("-" * 60)
    # 小目标
    print(f"{size_desc['small']:<20} {stats[3]:<10.4f} {stats[9]:<10.4f}")
    # 中目标
    print(f"{size_desc['medium']:<20} {stats[4]:<10.4f} {stats[10]:<10.4f}")
    # 大目标
    print(f"{size_desc['large']:<20} {stats[5]:<10.4f} {stats[11]:<10.4f}")
    # 整体对比
    print("-" * 60)
    print(f"{'整体目标':<20} {stats[1]:<10.4f} {stats[7]:<10.4f}")
 
 
def print_overall_summary(stats: np.ndarray, class_names: List[str]) -> None:
    """打印完整的COCO评估指标汇总（含所有IoU阈值和尺寸）"""
    metrics = [
        ("AP@[0.5:0.95]", "所有IoU阈值平均精度"),
        ("AP@0.5", "IoU=0.5平均精度"),
        ("AP@0.75", "IoU=0.75平均精度"),
        ("AP小目标", "IoU[0.5:0.95]小目标精度"),
        ("AP中目标", "IoU[0.5:0.95]中目标精度"),
        ("AP大目标", "IoU[0.5:0.95]大目标精度"),
        ("AR@[0.5:0.95]", "所有IoU阈值平均召回率"),
        ("AR@0.5", "IoU=0.5平均召回率"),
        ("AR@0.75", "IoU=0.75平均召回率"),
        ("AR小目标", "IoU[0.5:0.95]小目标召回率"),
        ("AR中目标", "IoU[0.5:0.95]中目标召回率"),
        ("AR大目标", "IoU[0.5:0.95]大目标召回率")
    ]
 
    print("\n" + "=" * 60)
    print(f"📋 完整评估结果汇总（检测类别：{', '.join(class_names)}）")
    print("=" * 60)
    for i, (metric_name, metric_desc) in enumerate(metrics):
        print(f"{metric_name:<15} | {metric_desc:<25} | {stats[i]:.4f}")
 
 
def evaluate_coco(pred_json: Path, anno_json: Path) -> np.ndarray:
    """核心评估逻辑：初始化COCO API、执行评估、输出结果"""
    print(f"\n[开始评估]")
    print(f"标注文件路径 → {anno_json}")
    print(f"预测文件路径 → {pred_json}")
 
    # 1. 先验证输入文件合法性
    for file in [pred_json, anno_json]:
        valid, msg = validate_json_file(file)
        if not valid:
            raise ValueError(f"文件验证失败：{msg}")
 
    try:
        start_time = time.time()
 
        # 2. 初始化COCO标注和预测API
        coco_anno = COCO(str(anno_json))  # 标注数据
        coco_pred = coco_anno.loadRes(str(pred_json))  # 预测数据（基于标注API加载）
 
        # 3. 获取数据集类别信息
        cat_ids = coco_anno.getCatIds()
        cat_info = coco_anno.loadCats(cat_ids)
        class_names = [cat["name"] for cat in cat_info]
 
        # 4. 执行bbox（边界框）评估
        coco_eval = COCOeval(coco_anno, coco_pred, "bbox")  # 指定评估类型为边界框
        coco_eval.evaluate()  # 计算评估指标
        coco_eval.accumulate()  # 累积统计结果
 
        # 5. 输出评估结果（分三部分：原始详细结果、IoU50专项、整体汇总）
        print("\n" + "-" * 80)
        print("🔍 pycocotools 原始评估结果（详细）")
        print("-" * 80)
        coco_eval.summarize()  # 打印pycocotools默认详细结果
 
        # 单独打印IoU50大中小目标指标（本次改进核心）
        print_iou50_size_metrics(coco_eval.stats)
 
        # 打印完整指标汇总
        print_overall_summary(coco_eval.stats, class_names)
 
        # 6. 计算评估耗时
        end_time = time.time()
        cost_time = end_time - start_time
        print(f"\n[评估完成] 总耗时：{cost_time:.2f}秒")
 
        return coco_eval.stats  # 返回完整统计结果数组，便于后续二次处理
 
    except Exception as e:
        raise RuntimeError(f"评估过程出错：{str(e)}") from e
 
 
def main():
    """命令行参数解析与主函数入口"""
    parser = argparse.ArgumentParser(
        description="COCO格式目标检测结果评估工具（支持IoU50大中小目标分析）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # 显示参数默认值
    )
    parser.add_argument('--annotations', type=str, default="/home/tgf/tgf/MRI/model/YOLO11_MRI/runs/detect/instances_val2017.json",
                        help='COCO格式的标注文件路径')
    parser.add_argument('--predictions', type=str, default="/home/tgf/tgf/MRI/model/YOLO11_MRI/runs/detect/val6_GCE/predictions.json",
                        help='预测结果的JSON文件路径')
    parser.add_argument(
        "--verbose", "-v", action="store_true", default=False,
        help="启用详细模式：显示更多评估过程日志（如类别数量、图像数量）"
    )
 
    args = parser.parse_args()
 
    # 转换为绝对路径，避免相对路径问题
    pred_json = Path(args.predictions).resolve()
    anno_json = Path(args.annotations).resolve()
 
    # 执行评估并捕获异常
    try:
        if args.verbose:
            # 详细模式：打印数据集基本信息
            coco_anno = COCO(str(anno_json))
            print(f"\n[详细信息] 数据集包含：")
            print(f"图像数量 → {len(coco_anno.getImgIds())}张")
            print(f"类别数量 → {len(coco_anno.getCatIds())}类")
 
        evaluate_coco(pred_json, anno_json)
    except Exception as e:
        print(f"\n❌ 评估失败：{str(e)}", file=sys.stderr)
        sys.exit(1)
 
 
if __name__ == "__main__":
    import sys  # 延迟导入，仅主函数需要
    main()