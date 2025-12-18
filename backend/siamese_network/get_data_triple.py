import os
import json
import random
import numpy as np
import pandas as pd
iris_path="../datasets/xinge_jpg/iris_fetures"
blood2fatures={}
fatures2blood={}
all_fatures=[]
with open("../datasets/xinge_jpg/datasetXGN/datasetXGN/blood.csv",'r') as f:
    all_data=f.readlines()

    for l in all_data:
        split=l.strip().split(',')
        blood=split[0]
        fatures=split[1:]
        for fa in fatures:
            if not os.path.exists(os.path.join(iris_path,fa+".jpg")):
                continue
            if blood not in blood2fatures.keys():
                blood2fatures[blood]=[]
            if fa not in fatures2blood.keys():
                fatures2blood[fa]=[]
            if fa not in blood2fatures[blood]:
                blood2fatures[blood].append(fa)
            if blood not in fatures2blood[fa]:
                fatures2blood[fa].append(blood)
            if fa not in all_fatures:
                all_fatures.append(fa)
print(f"共有{len(blood2fatures)}种血统")
print(f"平均每个信鸽有{sum([len(v) for _,v in fatures2blood.items()])/len(fatures2blood)}")
blood2generated_data={}
all_fatures=list(set(all_fatures))
all_tuple=set()
for blood,fs in blood2fatures.items():
    blood2generated_data[blood]=[]
    for index_f1 in range(len(fs)):
        avoid=[]
        f1=fs[index_f1]
        for b in fatures2blood[f1]:
            avoid.extend(blood2fatures[b])
        avoid=[temp for temp in avoid if temp !=f1]
        avoid=list(set(avoid))
        try:
            index=random.randint(0,len(avoid)-1)
        except:
            continue
        f2=avoid[index]
        negative=None
        while negative==None  :
            index=random.randint(0,len(all_fatures)-1)
            if all_fatures[index] not in avoid:
                negative=all_fatures[index]
                if tuple(sorted([f1, f2, negative])) in all_tuple:
                    negative=None
        all_tuple.add(tuple(sorted([f1, f2,negative])))
        blood2generated_data[blood].append({"anchor":f1, "positive":f2, "negative":negative})
print(f"total matches: {len(all_tuple)}")
all_blood=list(blood2fatures.keys())
random.shuffle(all_blood)
train_blood=all_blood[:int(len(all_blood) * 0.8)]
test_blood=all_blood[int(len(all_blood) * 0.8):int(len(all_blood) * 0.9)]
val_blood=all_blood[int(len(all_blood) * 0.9):]
def get_dataset(all_data,bloods,split_type):
    split_data=[]
    for b in bloods:
        split_data.extend(all_data[b])
    random.shuffle(split_data)
    print(f"血统数:{len(bloods)}")
    print(f"数据量:{len(split_data)}")
    with open(f"{split_type}_triple.json","w") as f:
        json.dump(split_data,f,indent=4)
get_dataset(blood2generated_data,train_blood,"train")
get_dataset(blood2generated_data,test_blood,"test")
get_dataset(blood2generated_data,val_blood,"val")