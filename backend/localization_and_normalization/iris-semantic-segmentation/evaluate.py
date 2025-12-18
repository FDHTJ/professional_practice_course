import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# --- 加载模型 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --- 推荐的轻量级模型 ---
model = smp.Unet(
    encoder_name="mobilenet_v2", # <--- 替换为 MobileNetV2
    encoder_weights=None,   # 同样使用预训练权重，加速收敛
    in_channels=3,
    classes=3,
).to(DEVICE)
model.load_state_dict(torch.load('../../backend/main/segmentation/pigeon_eye_segmentation_model.pth'))
model.to(DEVICE)
model.eval()

# --- 准备单张图片 ---
image_path = 'datasets/images/304457.jpg' # 换成你的图片路径
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

input_tensor = transform(image=image_rgb)['image'].unsqueeze(0).to(DEVICE)

# --- 推理 ---
with torch.no_grad():
    prediction = model(input_tensor)

# --- 后处理和可视化 ---
# prediction 的形状是 [1, 3, 256, 256]，我们需要找到每个像素点概率最大的类别
pred_mask = torch.argmax(prediction.squeeze(), dim=0).cpu().numpy().astype(np.uint8)

# 创建一个彩色的掩码用于可视化
# 背景=黑色, 瞳孔=蓝色, 虹膜=绿色
color_mask = np.zeros((256, 256, 3), dtype=np.uint8)
color_mask[pred_mask == 1] = [0, 0, 255] # 瞳孔 - 蓝色
color_mask[pred_mask == 2] = [0, 255, 0] # 虹膜 - 绿色

# 将掩码缩放回原始尺寸
original_h, original_w, _ = image.shape
color_mask_resized = cv2.resize(color_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

# 将掩码叠加到原始图片上
overlay = cv2.addWeighted(image, 0.7, color_mask_resized, 0.3, 0)

cv2.imwrite('output_overlay_1.jpg', overlay)
print("预测结果已保存为 output_overlay_1.jpg")
