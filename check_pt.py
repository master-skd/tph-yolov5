import torch

# 替换成你的 .pt 文件路径
weights_path = 'runs/train/tphv5l_lizheng_copyPaste/weights/best.pt' 

try:
    # 加载模型 (map_location='cpu' 确保即使没有 GPU 也能读取)
    ckpt = torch.load(weights_path, map_location='cpu')

    print("\n" + "="*30)
    print(f"正在检查模型: {weights_path}")
    print("="*30)

    # 1. 尝试读取 'opt' (旧版 YOLOv5) 或 'train_args' (新版 YOLOv5)
    # 这些字典里记录了 python train.py 运行时传入的所有参数
    train_opt = None
    if 'opt' in ckpt:
        train_opt = ckpt['opt']
        print("✅ 找到训练配置 (key: 'opt')")
    elif 'train_args' in ckpt:
        train_opt = ckpt['train_args']
        print("✅ 找到训练配置 (key: 'train_args')")
    
    if train_opt:
        # 获取 imgsz 参数
        imgsz = train_opt.get('imgsz', '未找到')
        print(f"🎯 训练分辨率 (imgsz): {imgsz}")
        
        # 顺便看看是不是使用了矩形训练 (Rectangular Training)
        rect = train_opt.get('rect', False)
        print(f"   矩形训练 (Rect): {rect}")
    else:
        print("⚠️ 未在 checkpoint 中找到训练配置信息 (opt/train_args)")

    # 2. 检查模型本身的最大步长 (Stride)
    # 这决定了分辨率必须是它的倍数 (通常是 32)
    if 'model' in ckpt:
        model = ckpt['model']
        if hasattr(model, 'stride'):
            max_stride = int(model.stride.max())
            print(f"📏 最大步长 (Max Stride): {max_stride}")
            print(f"   (输入分辨率应该是 {max_stride} 的倍数)")

    print("="*30 + "\n")

except Exception as e:
    print(f"❌ 读取失败: {e}")