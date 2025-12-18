import os
import json

import tqdm

print("图片总数量:", len(os.listdir("images")))
print("总标注数:",len(os.listdir("metaXG/metadataXG/anotations")))
anotation={}
all_anotations=[]
numbers={}
clearn_anotations=[]
with open("metaXG/metadataXG/anotations.json","r") as f:
    all_anotations=json.load(f)
    for d in all_anotations:
        # assert len(d["bbs"]) ==1
        if len(d["bbs"]) not in numbers.keys():
            numbers[len(d["bbs"])]=0
        numbers[len(d["bbs"])]+=1
        if len(d["bbs"])==1:
            for l in d["bbs"]:
                if l["label"]!="eye":
                    l["label"]="eye"
                if l["label"] not in anotation.keys():
                    anotation[l["label"]]=0
                anotation[l["label"]]+=1
            clearn_anotations.append(d)
with open("metaXG/metadataXG/clean_annotations.json","w") as f:
    print("len Clean Annotations:",len(clearn_anotations))
    json.dump(clearn_anotations,f, indent=4, ensure_ascii=False)
print(anotation)
print(numbers)
# for f in tqdm.tqdm(os.listdir("metaXG/metadataXG/anotations")):
#     d=json.load(open("metaXG/metadataXG/anotations/"+f))
#     all_anotations.append(d)
#     # for l in d["bbs"]:
#     #     if l["label"] not in anotation:
#     #         anotation.append(l["label"])
# # print(anotation)
# with open("metaXG/metadataXG/anotations.json","w") as f:
#     json.dump(all_anotations,f,indent=4,ensure_ascii=False)