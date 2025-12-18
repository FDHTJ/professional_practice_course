import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Segmentation:
    def __init__(self,state_file):
        # --- 加载模型 ---
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        # --- 推荐的轻量级模型 ---
        model = smp.Unet(
            encoder_name="mobilenet_v2",  # <--- 替换为 MobileNetV2
            encoder_weights=None,  # 同样使用预训练权重，加速收敛
            in_channels=3,
            classes=3,
        ).to(self.DEVICE)
        model.load_state_dict(torch.load(state_file,map_location=torch.device(self.DEVICE)))
        model.to(self.DEVICE)
        model.eval()
        self.model=model
    def get_normalize_iris(self,image_path,):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask=self.get_iris_and_pupil_mask(image_path,return_mask=True)
        normalized_iris=self.normalize_iris_direct(image,mask)
        return normalized_iris
    def get_iris_and_pupil_mask(self,image_path, save_path=None, return_mask=False):
        # --- 准备单张图片 ---
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        input_tensor = transform(image=image_rgb)['image'].unsqueeze(0).to(self.DEVICE)

        # --- 推理 ---
        with torch.no_grad():
            prediction = self.model(input_tensor)

        pred_mask = torch.argmax(prediction.squeeze(), dim=0).cpu().numpy().astype(np.uint8)

        original_h, original_w, _ = image.shape
        mask_resized = cv2.resize(pred_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

        if save_path is not None:
            cv2.imwrite(save_path, mask_resized)
        if return_mask:
            return mask_resized

    def get_mask_centroid(self,segmentation_mask, target_value=1):
        target_mask = np.where(segmentation_mask == target_value, 255, 0).astype(np.uint8)
        M = cv2.moments(target_mask)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return (cX, cY)
        else:
            # 如果区域不存在，返回None
            return None

    def normalize_iris_direct(self,gray_image, segmentation_mask,
                              normalized_height=64, normalized_width=512):

        normalized_iris = np.zeros((normalized_height, normalized_width), dtype=np.uint8)

        pupil_center = self.get_mask_centroid(segmentation_mask, target_value=1)
        if pupil_center is None:
            print("错误: 无法找到瞳孔质心。")
            return None

        max_radius = int(0.5 * np.sqrt(gray_image.shape[0] ** 2 + gray_image.shape[1] ** 2))

        for w in range(normalized_width):
            angle = (w / normalized_width) * 2 * np.pi
            ray_end_x = pupil_center[0] + max_radius * np.cos(angle)
            ray_end_y = pupil_center[1] + max_radius * np.sin(angle)
            x_points = np.linspace(pupil_center[0], ray_end_x, max_radius).astype(int)
            y_points = np.linspace(pupil_center[1], ray_end_y, max_radius).astype(int)
            p_inner, p_outer = None, None
            for x, y in zip(x_points, y_points):
                if 0 <= y < segmentation_mask.shape[0] and 0 <= x < segmentation_mask.shape[1]:
                    if segmentation_mask[y, x] == 2:  # 值为2的是虹膜
                        p_inner = (x, y)
                        break
            for x, y in reversed(list(zip(x_points, y_points))):
                if 0 <= y < segmentation_mask.shape[0] and 0 <= x < segmentation_mask.shape[1]:
                    if segmentation_mask[y, x] == 2:
                        p_outer = (x, y)
                        break
            if p_inner and p_outer:
                sample_x = np.linspace(p_inner[0], p_outer[0], normalized_height)
                sample_y = np.linspace(p_inner[1], p_outer[1], normalized_height)
                for h in range(normalized_height):
                    x, y = int(round(sample_x[h])), int(round(sample_y[h]))
                    if 0 <= y < gray_image.shape[0] and 0 <= x < gray_image.shape[1]:
                        normalized_iris[h, w] = gray_image[y, x]
        return normalized_iris