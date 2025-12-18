import json
import os
import shutil
import random

import tqdm


def get_image_and_labels(data:list,tag="train"):
    image_path="images/"
    image_target=f"../../yolov8n/images/{tag}/"
    label_target=f"../../yolov8n/labels/{tag}/"
    # 确保目标文件夹存在
    os.makedirs(image_target, exist_ok=True)
    os.makedirs(label_target, exist_ok=True)
    for d in tqdm.tqdm(data):
        assert len(d["bbs"])==1
        for l in d["bbs"]:
            shutil.copy(image_path+d["img"],image_target+d["img"])
            label_file_name=d["img"].split(".")[0]+".txt"
            w,h=d["weidth"],d["height"]
            x_center=(l["bbx"][0]+l["bbx"][2])/2/w
            y_center=(l["bbx"][1]+l["bbx"][3])/2/h
            width=(l["bbx"][2]-l["bbx"][0])/w
            height=(l["bbx"][3]-l["bbx"][1])/h
            with open(label_target+label_file_name,"w") as o:
                o.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

with open("metaXG/metadataXG/clean_annotations.json",'r') as f:
    datas=json.load(f)
    train_l=int(len(datas)*0.8)
    test_l=int(len(datas)*0.9)
    random.shuffle(datas)
    train=datas[:train_l]
    test=datas[train_l:test_l]
    val=datas[test_l:]
    print(len(train),len(test),len(val))
    get_image_and_labels(train,"train")
    get_image_and_labels(test,"test")
    get_image_and_labels(val,"val")
