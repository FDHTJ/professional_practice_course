from ultralytics import YOLO

# 1. 加载您训练好的模型
model_path = 'runs/detect/train3/weights/best.pt'  # 替换成您自己的 best.pt 路径
model = YOLO(model_path)

# 2. 指定数据集配置文件
data_config_path = 'data.yml'  # 您的数据集配置文件

# 3. 运行验证/评估
# 使用 model.val() 方法
metrics = model.val(
    data=data_config_path,
    split='test',  # 关键：指定使用测试集
    # imgsz=640,   # 可选，通常会自动使用训练时的尺寸
    # batch=16,    # 可选，根据您的显存调整
    # conf=0.25,   # 可选，评估时使用的置信度阈值
    # iou=0.7      # 可选，评估时使用的IoU阈值
)




# metrics 对象包含了所有详细信息
print(metrics)