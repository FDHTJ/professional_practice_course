# import sqlite3
# print(f"当前 SQLite 版本: {sqlite3.sqlite_version}")
import json
import os
os.environ["CHROMADB_DISABLE_RUST"] = "1"
import numpy as np
import pandas as pd
import pickle

from tqdm import tqdm


def is_pg_id(s):
    if s.find("-")!=-1 and s.find("--")==-1:
        return True
    contain_number=False
    contain_alphabet=False
    for c in s:
        if c.isdigit():
            contain_number=True
        if c.isalpha():
            contain_alphabet=True
    return contain_number and contain_alphabet
def load_pigeon_information():
    pigeon=[]
    keys=["ID","PID","CID","SID","NAME","PG_ID","IMG"]
    count=0
    with open("pigeon.csv", "r") as f:
        data=f.readlines()
        for l in data[1:]:
            values=l.split(",")
            item={}
            for i in range(5):
                item[keys[i]]=values[i]
            pg_id=""
            for v in values:
                if v.startswith("http"):
                    item["IMG"]=v
                    continue
                if is_pg_id(v):
                    pg_id=v

            item["PG_ID"]=pg_id
            pigeon.append(item)
    return pigeon


# load_pigeon_information()
def get_meta_information():
    with open("img2embedding.pkl", "rb") as f:
        img2embedding=pickle.load(f)
    all_reserve=list(img2embedding.keys())
    print("共有", len(all_reserve),"只鸽子")
    if os.path.exists("meta_data.json"):
        meta_data=json.load(open("meta_data.json"))
        return img2embedding,meta_data
    with open("city_list.json", "r") as f:
        ids2city = {}
        data = json.load(f)
        for d in data:
            ids2city[d["pid"]]={}
            ids2city[d["pid"]]["name"] = d["pname"]
            ids2city[d["pid"]]["cid2cname"] = {}
            for c in d["city"]:
                ids2city[d["pid"]]["cid2cname"][c[0]]=c[1]
    with open("details.txt", "r") as f:
        detail_lines = f.readlines()
        details={}
        for line in detail_lines:
            l=line.strip().split("\t")
            # assert len(l)==2
            if len(l)<2:continue
            details[l[0]]="".join(l[1])
    meta_data={}

    pigeon=load_pigeon_information()#pd.read_csv("pigeon.csv",on_bad_lines='skip')
    count=0
    for row in pigeon:
        for k in row.keys():
            if not isinstance(row[k],str):
                row[k]=str(row[k])
        if row["ID"] not in all_reserve:
            continue
        count+=1
        meta_data[row["ID"]]={
            "name":row["NAME"],
            "province":ids2city[row["PID"]]["name"],
            "city":ids2city[row["PID"]]["cid2cname"][row["CID"]],
            # "color":row["COLOR"],
            # "eye":row["EYE"],
            "pg_id":row["PG_ID"],
            # "gender":row["SEX"],
            # "blood":row["BLOOD"],
            "image":row["IMG"],
            "details":details[row["ID"]],
        }
    print(count)
    for k,v in img2embedding.items():
        img2embedding[k]=np.array(v, dtype=np.float32).tolist()
    return img2embedding,meta_data

embeddings,meta_data = get_meta_information()
ids=list(embeddings.keys())
embeddings=[embeddings[id].tolist() for id in ids]
meta_data=[meta_data[id] for id in ids]
# for m in meta_data:
#     m.pop("details")
import chromadb

# 1. 初始化客户端 (这里演示本地持久化模式)
client = chromadb.PersistentClient(path="./pigeon_db")

# 2. 创建集合
# 注意：因为你自己提供向量，这里不需要 embedding_function
collection = client.get_or_create_collection(name="pigeon",embedding_function=None)
# documents=[meta_data[id].pop("details") for id in ids]
batch_size=500
for i in tqdm( range(0,len(ids),batch_size)):
    batch_ids=ids[i:i+batch_size]
    batch_meta=meta_data[i:i+batch_size]
    batch_embeddings=embeddings[i:i+batch_size]
    # 3. 写入数据
    # 核心点：显式传入 embeddings 参数
    collection.add(
        ids=batch_ids,
        embeddings=batch_embeddings,
        metadatas=batch_meta,
    )

print(f"成功写入 {collection.count()} 条数据")
