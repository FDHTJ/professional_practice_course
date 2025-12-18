import shutil
import os

dst_folder = "images"  # 目标文件夹

def move_files(index):
    src_folder = f"{index}/{index}"   # 源文件夹

    # 确保目标文件夹存在
    os.makedirs(dst_folder, exist_ok=True)

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(src_folder):
        src_path = os.path.join(src_folder, filename)
        dst_path = os.path.join(dst_folder, filename)

        # 只移动文件（不包括文件夹）
        if os.path.isfile(src_path):
            shutil.move(src_path, dst_path)
            print(f"✅ 已移动: {filename}")
for i in range(1,13):
    move_files(i)
