import os
import shutil
import cv2
import numpy as np

IMAGE_PATH="extract_eyes_by_yolo"
MASK_PATH="segmentation_masks"
target_path="failed"
with open("all_failed_images.txt",'r') as f:
    lines=f.readlines()
    for l in lines:
        l=l.strip()
        image=cv2.imread(os.path.join(IMAGE_PATH,l))
        mask=cv2.imread(os.path.join(MASK_PATH,l.replace(".jpg",".png")),cv2.IMREAD_GRAYSCALE)
        original_h, original_w, _ = image.shape
        color_mask = np.zeros((original_h, original_w, 3), dtype=np.uint8)
        color_mask[mask == 1] = [0, 0, 255]  # 瞳孔 - 蓝色
        color_mask[mask == 2] = [0, 255, 0]  # 虹膜 - 绿色
        # color_mask_resized = cv2.resize(color_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        overlay = cv2.addWeighted(image, 0.7, color_mask, 0.3, 0)

        cv2.imwrite(os.path.join(target_path,l), overlay)
        # break
