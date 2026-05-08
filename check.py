import onnx

try:
    # 注意：这里加载的是 bestC.onnx (原始文件)，不是 sim
    model = onnx.load("runs/train/tphv5l_lizheng_copyPaste/weights/best.onnx")
    
    print("\n=== 原始模型输入检查 ===")
    for input in model.graph.input:
        shape = []
        for d in input.type.tensor_type.shape.dim:
            # 如果是动态维度，打印 "?"，否则打印数值
            if d.dim_param:
                shape.append(f"? ({d.dim_param})")
            elif d.dim_value > 0:
                shape.append(d.dim_value)
            else:
                shape.append("?")
        print(f"节点名称: {input.name}")
        print(f"节点形状: {shape}")

except Exception as e:
    print(f"读取失败: {e}")