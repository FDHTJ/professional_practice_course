import os
import json
import pickle
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.optim as optim
from tqdm import tqdm
class BaseFeatureNet(nn.Module):
    def __init__(self):
        super(BaseFeatureNet, self).__init__()
        # 输入形状: (N, 1, 64, 512)
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 32x256

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> 16x128

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2)  # -> 8x64
        )
        self.fc_block = nn.Sequential(
            nn.Linear(128 * 8 * 64, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)  # 输出128维的特征向量 (Embedding)
        )
    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc_block(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.base_net = BaseFeatureNet()

    def forward_one(self, x):
        # 用于推理时，提取单张图片的特征
        return self.base_net(x)

    def forward(self, input):
        # 用于训练时，提取一对图片的特征
        output = self.base_net(input)

        return output
class Siamese:
    def __init__(self,state_file):
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"当前使用的设备是: {self.DEVICE}")
        transform = transforms.Compose([
            transforms.ToTensor(),  # 将图片转换为Tensor，并归一化到[0,1]
            transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化到[-1,1]，适用于灰度图
        ])
        # --- 3. 初始化模型、损失函数和优化器 ---
        model = SiameseNetwork().to(self.DEVICE)
        model.eval()
        model.load_state_dict(torch.load(state_file, map_location=self.DEVICE))
        self.model = model
        self.transform = transform
    def process_image(self,image_path):
        image = Image.open(image_path).convert('L')
        image = self.transform(image)
        output = self.model(image.unsqueeze(0).to(self.DEVICE))
        embedding = output.flatten().detach().cpu()
        return embedding

    def is_one_blood(self, image_path1, image_path2):
        embedding1 = self.process_image(image_path1)
        embedding2 = self.process_image(image_path2)
        d = F.pairwise_distance(embedding1, embedding2).item()
        if d<=0.9325207:
            return (True,d)
        else:
            return (False,d)

    def query(self, image_path, top_k,database_path):
        embedding = self.process_image(image_path).tolist()
        import chromadb

        client = chromadb.PersistentClient(path=database_path)
        collection = client.get_collection(name="pigeon")
        print("embedding length:", len(embedding))
        # stats = collection.get(limit=5)
        # print(stats)

        results = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
        )
        return results["metadatas"][0]
