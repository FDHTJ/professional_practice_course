import json
import os

import tqdm
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
def show_image_with_bbs(image_path:str,bbs:list,confidence=None):
    # 读取图像
    image = cv2.imread(image_path)

    # 将 BGR 转换为 RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 创建一个图形窗口
    fig, ax = plt.subplots()

    # 显示图像
    ax.imshow(image)

    # 定义边框的坐标 (左上角 x, 左上角 y, 宽度, 高度)
    if len(bbs) > 0:
        for bbox in bbs:
        # 创建一个矩形边框
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none')
            # ax.text(bbox[0], bbox[1], str(confidence), fontsize=9,
            #     ha='left', va='bottom', color="red" # 左上角对齐
            #     # transform=ax.transAxes       # 若坐标是 0~1 的轴坐标
            #     )
        # 添加矩形到图像上
            ax.add_patch(rect)


    # 显示图像
    plt.show()
# 1. 加载您训练好的模型
model_path = 'runs/detect/train3/weights/best.pt'  # 替换成您自己的 best.pt 路径
model = YOLO(model_path)

# 2. 指定数据集配置文件
data_config_path = 'data.yml'  # 您的数据集配置文件

# 3. 运行验证/评估
# 使用 model.val() 方法
count={}
failed_images=[]
images="../datasets/xinge_jpg/images"
output_idr="../datasets/xinge_jpg/extract_eyes_by_yolo"
numbers=[]
for image in tqdm.tqdm(os.listdir(images)):
    if os.path.exists(output_idr+"/"+image):
        continue
    try:
        pred = model.predict(source=os.path.join(images,image),conf=0.439)
    except Exception as e:
        failed_images.append(image)
        continue
    p=pred[0].boxes
    if len(p)==1:
        original_img = pred[0].orig_img
        xyxy=p.xyxy.flatten().tolist()
        x_min,y_min,x_max,y_max =[int(i) for i in xyxy]# xyxy[0],xyxy[1],xyxy[2],xyxy[3]
        cropped_img = original_img[y_min:y_max, x_min:x_max]
        cv2.imwrite(os.path.join(output_idr,image), cropped_img)
    else:
        if len(p) not in numbers:
            show_image_with_bbs(os.path.join(images,image),p.xyxy.tolist())
            numbers.append(len(p))
        if len(p) not in count.keys():
            count[len(p)]=0
        count[len(p)]+=1
print(count)

# with open("predict_results/count.json",'w') as f:
#     json.dump(count,f,indent=4,sort_keys=True)
#
# with open("predict_results/failed_images.txt",'w') as f:
#     for image in failed_images:
#         f.write(image+"\n")

