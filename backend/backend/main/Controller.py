from fastapi import FastAPI, UploadFile, File
import shutil
import os

import cv2
from siamese.Siamese import Siamese
from segmentation.Segmentation import Segmentation
def initialize():
    seg=Segmentation("segmentation/pigeon_eye_segmentation_model.pth")
    sia=Siamese("siamese/best_siamese_model_triple.pth")
    return seg,sia
seg,sia=initialize()
app = FastAPI()

# 创建一个文件夹存上传的图
UPLOAD_DIR = "uploaded_eyes"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/upload/eye")
async def upload_eye(file_1: UploadFile = File(...), file_2: UploadFile = File(...)):
    # 1. 构造保存路径
    file1_location = f"{UPLOAD_DIR}/{file_1.filename}"
    file2_location = f"{UPLOAD_DIR}/{file_2.filename}"
    # 2. 保存文件到本地
    with open(file1_location, "wb") as buffer:
        shutil.copyfileobj(file_1.file, buffer)
    with open(file2_location, "wb") as buffer:
        shutil.copyfileobj(file_2.file, buffer)
    print(f"接收到图片: {file_1.filename}")
    print(f"接收到图片: {file_2.filename}")
    normalized_iris1=seg.get_normalize_iris(file1_location)
    normalized_iris2=seg.get_normalize_iris(file2_location)
    if normalized_iris1 is None or normalized_iris2 is None:
        return {"status": "fail", "message":"服务器端错误，请更换图片后重试","data":{}}
    normal1_location=f"{UPLOAD_DIR}/normal_{file_1.filename}"
    normal2_location=f"{UPLOAD_DIR}/normal_{file_2.filename}"
    cv2.imwrite(normal1_location, normalized_iris1)
    cv2.imwrite(normal2_location, normalized_iris2)
    is_one_blood,d=sia.is_one_blood(normal1_location,normal2_location)
    for f in [file1_location,file2_location,normal1_location,normal2_location]:
        os.remove(f)
    # 3. 返回成功信息
    return {"status": "success", "message": "虹膜接收成功","data":{"result": ("它们属于同一血统," if is_one_blood else "它们不属于同一血统,")+f"特征向量距离为{d:.2f}"}}
@app.post("/upload/eye_retrieval")
async def upload_eye_retrieval(file: UploadFile = File(...)):
    # 1. 构造保存路径
    file_location = f"{UPLOAD_DIR}/{file.filename}"

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print(f"接收到图片: {file.filename}")
    normalized_iris=seg.get_normalize_iris(file_location)
    if normalized_iris is None:
        return {"status": "fail", "message":"服务器端错误，请更换图片后重试","data":{}}
    normal_location=f"{UPLOAD_DIR}/normal_{file.filename}"
    cv2.imwrite(normal_location, normalized_iris)
    query_result=sia.query(normal_location,top_k=50,database_path="../database/pigeon_db")
    for f in [file_location,normal_location]:
        os.remove(f)
    # 3. 返回成功信息
    return {"status": "success", "message": "虹膜接收成功","data":{"result": query_result}}


if __name__ == "__main__":
    # 注意：host="0.0.0.0" 允许局域网访问
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)