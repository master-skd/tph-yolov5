import os
os.environ['WANDB_MODE'] = 'disabled'

import argparse
import shutil
from pathlib import Path

# 导入现有的在线训练逻辑
import train_online


def run_all():
    parser = argparse.ArgumentParser()

    # 路径参数
    parser.add_argument(
        '--train-dir',
        type=str,
        default='/data1/code/xy/datasets/Anti-UAV-video/train',
        help='训练集视频根目录'
    )
    parser.add_argument(
        '--object-dir',
        type=str,
        default='extracted_objects',
        help='提取的无人机目标库'
    )
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='初始权重路径'
    )

    # 训练超参
    parser.add_argument('--epochs', type=int, default=3, help='每个视频微调的轮数')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--freeze', type=str, default='10', help='冻结层数 (默认冻结backbone)')

    # 输出参数
    parser.add_argument('--project', type=str, default='runs/online_train_all', help='保存项目名')
    parser.add_argument('--name', type=str, default='sequential_finetune', help='保存实验名')
    parser.add_argument('--keep-last', type=int, default=3, help='只保留最近多少个step目录')

    opt = parser.parse_args()

    train_dir = Path(opt.train_dir)
    if not train_dir.exists():
        print(f"训练目录不存在: {train_dir}")
        return

    weights_path = Path(opt.weights)
    if not weights_path.exists():
        print(f"初始权重不存在: {weights_path}")
        return

    # 1. 搜集所有包含 visible.mp4 的视频目录
    video_folders = []
    for sub_dir in train_dir.iterdir():
        if sub_dir.is_dir() and (sub_dir / 'visible.mp4').exists():
            video_folders.append(sub_dir)

    # 可选：排序，确保处理顺序稳定
    video_folders = sorted(video_folders, key=lambda x: x.name)

    print(f"找到 {len(video_folders)} 个训练视频序列。")

    if len(video_folders) == 0:
        print("没有找到可用的视频目录（需包含 visible.mp4）。")
        return

    current_weights = str(weights_path)
    project_dir = Path(opt.project)
    project_dir.mkdir(parents=True, exist_ok=True)

    # 记录成功训练后保留的 step 目录
    kept_step_dirs = []

    for i, folder in enumerate(video_folders):
        print(f"\n[{i+1}/{len(video_folders)}] 正在处理视频: {folder.name}")

        video_path = str(folder / 'visible.mp4')
        temp_data_dir = f"temp_data_{folder.name}"

        # A. 生成合成数据集
        dataset_yaml = train_online.generate_online_dataset(
            video_path=video_path,
            object_dir=opt.object_dir,
            output_dir=temp_data_dir,
            num_frames=None,       # 使用全部帧
            objects_per_frame=3
        )

        if not dataset_yaml:
            print(f"视频 {folder.name} 数据生成失败，跳过。")
            if os.path.exists(temp_data_dir):
                shutil.rmtree(temp_data_dir, ignore_errors=True)
            continue

        # B. 配置训练选项
        import train
        train_opt = train.parse_opt(known=True)
        train_opt.data = dataset_yaml
        train_opt.weights = current_weights
        train_opt.epochs = opt.epochs
        train_opt.batch_size = opt.batch_size
        train_opt.imgsz = opt.imgsz
        train_opt.project = opt.project
        train_opt.name = f"{opt.name}_step_{i}_{folder.name}"
        train_opt.exist_ok = True

        # 处理冻结
        if ',' in str(opt.freeze):
            train_opt.freeze = [int(x) for x in str(opt.freeze).split(',')]
        else:
            train_opt.freeze = int(opt.freeze)

        step_dir = Path(train_opt.project) / train_opt.name
        last_best = step_dir / 'weights' / 'best.pt'

        # C. 执行训练
        try:
            train.main(train_opt)

            # 更新权重路径为最新生成的权重，供下一个视频使用
            if last_best.exists():
                current_weights = str(last_best)
                print(f"完成。权重已更新为: {current_weights}")

                # 记录当前 step 目录
                kept_step_dirs.append(step_dir)

                # 只保留最近 keep-last 个 step
                while len(kept_step_dirs) > opt.keep_last:
                    old_dir = kept_step_dirs.pop(0)

                    # 删除旧目录
                    if old_dir.exists():
                        print(f"删除旧step目录: {old_dir}")
                        shutil.rmtree(old_dir, ignore_errors=True)
            else:
                print(f"警告: 未找到生成的权重 {last_best}，将继续使用旧权重。")

        except Exception as e:
            print(f"训练视频 {folder.name} 时出错: {e}")

        finally:
            # 清理临时数据以节省空间
            if os.path.exists(temp_data_dir):
                shutil.rmtree(temp_data_dir, ignore_errors=True)

    # 额外保存最终权重到固定位置
    final_save = project_dir / 'final_best.pt'
    try:
        if Path(current_weights).exists():
            shutil.copy2(current_weights, final_save)
            print(f"\n所有视频在线学习完成！")
            print(f"最终权重保存在: {current_weights}")
            print(f"最终权重额外复制到: {final_save}")
        else:
            print(f"\n所有视频在线学习完成！但最终权重不存在: {current_weights}")
    except Exception as e:
        print(f"\n所有视频在线学习完成，但复制最终权重失败: {e}")
        print(f"最终权重原路径仍为: {current_weights}")


if __name__ == "__main__":
    run_all()