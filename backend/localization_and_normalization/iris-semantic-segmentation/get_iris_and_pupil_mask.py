import os

import torch
import cv2
import numpy as np
import albumentations as A
import tqdm
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
def get_iris_and_pupil_mask(image_path,save_path=None,return_mask=False):
    # --- 准备单张图片 ---
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

    pred_mask = torch.argmax(prediction.squeeze(), dim=0).cpu().numpy().astype(np.uint8)


    original_h, original_w, _ = image.shape
    mask_resized = cv2.resize(pred_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    if save_path is not None:
        cv2.imwrite(save_path, mask_resized)
    if return_mask:
        return mask_resized

IMAGE_DIR="../../datasets/xinge_jpg/extract_eyes_by_yolo"
MASK_DIR="../../datasets/xinge_jpg/segmentation_masks"
for f in tqdm.tqdm(os.listdir(IMAGE_DIR)):
    get_iris_and_pupil_mask(os.path.join(IMAGE_DIR,f),os.path.join(MASK_DIR,f.replace(".jpg",".png")))
