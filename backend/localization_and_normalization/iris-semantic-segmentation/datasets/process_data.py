import shutil
import os
import random

import cv2
import numpy as np


def get_datas():
    images=os.listdir("../../../datasets/xinge_jpg/extract_eyes_by_yolo")
    random.shuffle(images)
    output="images"
    images=images[:200]
    for image in images:
        shutil.copy(os.path.join("../../../datasets/xinge_jpg/extract_eyes_by_yolo", image),os.path.join(output,image))

def get_train_data():
    with open("image_file_2_task_id.json",'r') as f:
        import json
        i2t=json.load(f)
        file2id={}
        for i in i2t:
            file=i["image"].split("%5C")[-1]
            id=i["id"]
            file2id[file]=id
    source="images"
    target="labeled_images"
    for k,v in file2id.items():
        shutil.copy(os.path.join(source,k),os.path.join(target,f"{v}.jpg"))

def process_mask():
    masks_dir="masks"
    files=os.listdir(masks_dir)
    for i in range(1,66):
        iris_file=[f for f in files if f.find(f"task-{i}-")!=-1 and f. find("iris")!=-1]
        #f"task-{i}-annotation-{i if i <10 else i+1}-by-1-label-iris-0.png"
        pupil_file=[f for f in files if f.find(f"task-{i}-")!=-1 and f. find("pupil")!=-1]#f"task-{i}-annotation-{i if i <10 else i+1}-by-1-label-pupil-0.png"

        assert len(iris_file)==len(pupil_file)==1
        iris_file=iris_file[0]
        pupil_file=pupil_file[0]
        iris_mask = cv2.imread(os.path.join(masks_dir, iris_file), cv2.IMREAD_GRAYSCALE)
        pupil_mask = cv2.imread(os.path.join(masks_dir, pupil_file), cv2.IMREAD_GRAYSCALE)

        # 初始化一个和原图一样大的空白掩码 (默认值为0，代表背景)
        height, width = iris_mask.shape
        combined_mask = np.zeros((height, width), dtype=np.uint8)

        # 合并掩码：先画虹膜(2)，再画瞳孔(1)，因为瞳孔在虹膜之上
        # 将掩码中白色(255)的区域赋值为我们的类别ID
        combined_mask[iris_mask == 255] = 2  # 虹膜类别为2
        combined_mask[pupil_mask == 255] = 1  # 瞳孔类别为1

        # 保存合并后的掩码
        output_dir="processed_masks"
        output_filename = f"{i}.png"
        cv2.imwrite(os.path.join(output_dir, output_filename), combined_mask)
process_mask()
