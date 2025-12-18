import os

import cv2
import numpy as np
import tqdm


def get_mask_centroid(segmentation_mask, target_value=1):
    """计算掩码中特定值区域的质心。"""
    # 创建一个二值图像，目标区域为255
    target_mask = np.where(segmentation_mask == target_value, 255, 0).astype(np.uint8)

    # 计算图像的矩 (Moments)
    M = cv2.moments(target_mask)

    # 从矩中计算质心坐标
    # 为了防止除以0，需要检查 m00
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)
    else:
        # 如果区域不存在，返回None
        return None


def normalize_iris_direct(gray_image, segmentation_mask,
                          normalized_height=64, normalized_width=512):
    """
    [直接法] 使用像素级掩码信息进行虹膜展开。
    """
    normalized_iris = np.zeros((normalized_height, normalized_width), dtype=np.uint8)

    # 1. 获取瞳孔质心作为射线的原点
    pupil_center = get_mask_centroid(segmentation_mask, target_value=1)
    if pupil_center is None:
        print("错误: 无法找到瞳孔质心。")
        return None  # 返回一个空图像

    # 确定射线的最大搜索半径，图像对角线的一半就足够了
    max_radius = int(0.5 * np.sqrt(gray_image.shape[0] ** 2 + gray_image.shape[1] ** 2))

    # 2. 遍历每一个角度 (对应归一化图像的每一列)
    for w in range(normalized_width):
        angle = (w / normalized_width) * 2 * np.pi

        # 3. "发射射线"：从质心开始，沿着角度方向向外搜索
        # 我们通过在一个很长的线段上采样点来实现
        ray_end_x = pupil_center[0] + max_radius * np.cos(angle)
        ray_end_y = pupil_center[1] + max_radius * np.sin(angle)

        # 使用np.linspace获取射线上的所有整数坐标点
        x_points = np.linspace(pupil_center[0], ray_end_x, max_radius).astype(int)
        y_points = np.linspace(pupil_center[1], ray_end_y, max_radius).astype(int)

        p_inner, p_outer = None, None

        # 4. 寻找内外边界
        # 从质心向外走，寻找第一个虹膜像素
        for x, y in zip(x_points, y_points):
            if 0 <= y < segmentation_mask.shape[0] and 0 <= x < segmentation_mask.shape[1]:
                if segmentation_mask[y, x] == 2:  # 值为2的是虹膜
                    p_inner = (x, y)
                    break

        # 从最远处向内走，寻找第一个虹膜像素
        for x, y in reversed(list(zip(x_points, y_points))):
            if 0 <= y < segmentation_mask.shape[0] and 0 <= x < segmentation_mask.shape[1]:
                if segmentation_mask[y, x] == 2:
                    p_outer = (x, y)
                    break

        # 5. 如果成功找到了内外边界
        if p_inner and p_outer:
            # 在内外边界点之间等距采样
            sample_x = np.linspace(p_inner[0], p_outer[0], normalized_height)
            sample_y = np.linspace(p_inner[1], p_outer[1], normalized_height)

            # 6. 提取像素值并填充到归一化图像的列中
            for h in range(normalized_height):
                x, y = int(round(sample_x[h])), int(round(sample_y[h]))
                if 0 <= y < gray_image.shape[0] and 0 <= x < gray_image.shape[1]:
                    normalized_iris[h, w] = gray_image[y, x]

    return normalized_iris

IMAGE_PATH="extract_eyes_by_yolo"
MASK_PATH="segmentation_masks"
FETURES_PATH= "iris_fetures"
failed = []

for f in tqdm.tqdm(os.listdir(IMAGE_PATH)):
    image=cv2.imread(os.path.join(IMAGE_PATH, f), cv2.IMREAD_GRAYSCALE)
    mask=cv2.imread(os.path.join(MASK_PATH, f.replace(".jpg",".png")),cv2.IMREAD_GRAYSCALE)
    normalized_iris = normalize_iris_direct(image, mask)
    if normalized_iris is not None:
        cv2.imwrite(os.path.join(FETURES_PATH, f), normalized_iris)
    else:
        failed.append(f)
        print("failed:",f)
print(f"完成，共{len(failed)}条失败")
with open("all_failed_images.txt", "w") as f:
    f.write("\n".join(failed))

