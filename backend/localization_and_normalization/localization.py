
import cv2
import numpy as np
import matplotlib.pyplot as plt


def localize_and_normalize_iris(image_path):
    """
    加载虹膜图像，执行定位、边界检测和归一化展开。

    参数:
    image_path (str): 虹膜图像的文件路径。
    """
    # 以灰度模式读取原始图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"错误: 无法加载图像，请检查路径: {image_path}")
        return

    # 去除高频噪声
    blurred_img = cv2.medianBlur(img, 3)

    # --- 步骤 S1: 瞳孔位置的粗略估计 ---
    # 通过一维投影找到强度最低的区域，该区域近似对应于瞳孔中心。
    # 这对应于图片中的公式：x = argmin(sum(I(x,y))) 和 y = argmin(sum(I(x,y)))
    vertical_projection = np.sum(blurred_img, axis=0)  # 垂直投影（沿列求和）
    horizontal_projection = np.sum(blurred_img, axis=1)  # 水平投影（沿行求和）

    coarse_pupil_x = np.argmin(vertical_projection)
    coarse_pupil_y = np.argmin(horizontal_projection)

    print(f"S1 - 粗定位瞳孔中心: ({coarse_pupil_x}, {coarse_pupil_y})")

    # --- 步骤 S2: 使用霍夫变换精确定位虹膜内外边界 ---
    # 在粗定位的瞳孔中心附近的一定区域内，采用边缘检测(Canny)和霍夫变换相结合的方法。
    # cv2.HoughCircles 内部集成了边缘检测。

    # 1. 检测内圆（瞳孔边界）
    # 参数需要根据图像集进行微调
    # dp: 累加器分辨率与图像分辨率的反比。
    # minDist: 检测到的圆心之间的最小距离。
    # param1: Canny边缘检测的高阈值。
    # param2: 圆心累加器的阈值，越小检测到的圆越多。
    # minRadius, maxRadius: 检测圆的最小和最大半径。
    pupil_circles = cv2.HoughCircles(blurred_img, cv2.HOUGH_GRADIENT, dp=1, minDist=200,
                                     param1=80, param2=50,
                                     minRadius=20, maxRadius=300)

    if pupil_circles is None:
        print("S2 - 错误: 未能检测到瞳孔边界。")
        return

    # 提取最可能的圆（通常是第一个）作为瞳孔
    pupil_params = np.uint16(np.around(pupil_circles[0, 0]))
    pupil_center = (pupil_params[0], pupil_params[1])
    pupil_radius = pupil_params[2]
    print(f"S2 - 精确定位瞳孔: 中心={pupil_center}, 半径={pupil_radius}")

    # 2. 检测外圆（虹膜边界）
    # 通常虹膜边界对比度较低，需要调整参数
    iris_circles = cv2.HoughCircles(blurred_img, cv2.HOUGH_GRADIENT, dp=1, minDist=200,
                                    param1=50, param2=10,
                                    minRadius=pupil_radius + 150, maxRadius=pupil_radius + 200)

    if iris_circles is None:
        print("S2 - 错误: 未能检测到虹膜边界。")
        return

    # 提取虹膜边界
    iris_params = np.uint16(np.around(iris_circles[0, 0]))
    # 理论上虹膜和瞳孔同心，因此使用瞳孔的中心作为虹膜中心以提高精度
    iris_center = pupil_center
    iris_radius = iris_params[2]
    print(f"S2 - 精确定位虹膜: 中心={iris_center}, 半径={iris_radius}")

    # --- 虹膜归一化展开 (Daugman's Rubber Sheet Model) ---
    # 将定位出的环状虹膜区域转换为一个固定大小的矩形
    height, width = 64, 512  # 归一化后的标准尺寸
    normalized_iris = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            # 从归一化极坐标 (i, j) 映射回原始笛卡尔坐标 (x, y)
            theta = (2 * np.pi * j) / width  # 角度
            rho = pupil_radius + ((iris_radius - pupil_radius) * i) / height  # 半径

            # 计算原始图像中的坐标
            x = int(iris_center[0] + rho * np.cos(theta))
            y = int(iris_center[1] + rho * np.sin(theta))

            # 确保坐标在图像范围内
            if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                # 使用最近邻插值获取像素值
                normalized_iris[i, j] = img[y, x]

    # --- 结果可视化 ---
    # 绘制定位结果
    img_with_circles = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.circle(img_with_circles, pupil_center, pupil_radius, (0, 255, 0), 2)  # 绿色圆: 瞳孔
    cv2.circle(img_with_circles, iris_center, iris_radius, (0, 0, 255), 2)  # 红色圆: 虹膜

    # 使用 Matplotlib 显示
    plt.figure(figsize=(18, 5))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(img_with_circles)
    plt.title('Location (S2)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(normalized_iris, cmap='gray')
    plt.title('Normalized Iris')
    plt.axis('off')

    plt.show()


# --- 主程序入口 ---
if __name__ == "__main__":
    # 将 'sample_iris.png' 替换为你的虹膜图像文件路径
    image_file = '../datasets/xinge_jpg/extract_eyes_by_yolo/100010.jpg'
    localize_and_normalize_iris(image_file)