import os
import json

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.optim as optim
from tqdm import tqdm


# ==============================================================================
# 步骤 1: 创建自定义的 Dataset 类来加载您的数据
# ==============================================================================
class SiameseIrisDataset(Dataset):
    """
    用于加载虹膜图像对的自定义数据集。
    """

    def __init__(self, json_file_path, features_dir, transform=None):
        # 1. 加载JSON文件
        with open(json_file_path, 'r') as f:
            self.image_pairs_list = json.load(f)

        self.features_dir = features_dir
        self.transform = transform
        self.file_extension = ".jpg"  # 确保这与您的特征图文件扩展名一致

    def __len__(self):
        return len(self.image_pairs_list)

    def __getitem__(self, index):
        # 1. 获取一对数据
        pair_data = self.image_pairs_list[index]
        pigeon1_id = pair_data["pigeon1"]
        pigeon2_id = pair_data["pigeon2"]
        label = pair_data["label"]

        # 2. 构建图像的完整路径
        img1_path = os.path.join(self.features_dir, pigeon1_id + self.file_extension)
        img2_path = os.path.join(self.features_dir, pigeon2_id + self.file_extension)

        # 3. 加载图像 (以单通道灰度图 'L' 模式)
        img1 = Image.open(img1_path).convert('L')
        img2 = Image.open(img2_path).convert('L')

        # 4. 应用数据变换
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # 5. 返回一对图像和它们的标签
        return img1, img2, torch.tensor(label, dtype=torch.float32)

class SiameseIrisDatasetTriple(Dataset):
    """
    用于加载虹膜图像对的自定义数据集。
    """

    def __init__(self, json_file_path, features_dir, transform=None):
        # 1. 加载JSON文件
        with open(json_file_path, 'r') as f:
            self.image_pairs_list = json.load(f)

        self.features_dir = features_dir
        self.transform = transform
        self.file_extension = ".jpg"  # 确保这与您的特征图文件扩展名一致

    def __len__(self):
        return len(self.image_pairs_list)

    def __getitem__(self, index):
        # 1. 获取一对数据
        pair_data = self.image_pairs_list[index]
        anchor = pair_data["anchor"]
        positive = pair_data["positive"]
        negative = pair_data["negative"]

        # 2. 构建图像的完整路径
        anchor_path = os.path.join(self.features_dir, anchor + self.file_extension)
        positive_path = os.path.join(self.features_dir, positive + self.file_extension)
        negative_path = os.path.join(self.features_dir, negative + self.file_extension)

        # 3. 加载图像 (以单通道灰度图 'L' 模式)
        anchor = Image.open(anchor_path).convert('L')
        positive = Image.open(positive_path).convert('L')
        negative = Image.open(negative_path).convert('L')

        # 4. 应用数据变换
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        # 5. 返回一对图像和它们的标签
        return anchor, positive, negative

# ==============================================================================
# 步骤 2: 构建孪生网络的模型架构
# ==============================================================================
class BaseFeatureNet(nn.Module):
    """
    基础的CNN网络，用于从单张虹膜图中提取特征向量。
    这个网络将被孪生网络的两个分支共享。
    """

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

        # 展平后的维度: 128 * 8 * 64 = 65536
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
        # L2归一化，使得特征向量都在一个单位超球面上，有助于稳定距离度量
        x = F.normalize(x, p=2, dim=1)
        return x


class SiameseNetwork(nn.Module):
    """
    孪生网络封装。
    """

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


# ==============================================================================
# 步骤 3: 定义对比损失函数
# ==============================================================================

# ==============================================================================
# 步骤 4: 完整的训练和验证流程
# ==============================================================================
if __name__ == '__main__':

    # --- 1. 配置参数 ---
    TRAIN_JSON_PATH = "train.json"
    VAL_JSON_PATH = "test_triple.json"
    FEATURES_DIR = "../datasets/xinge_jpg/iris_fetures"  # 您的虹膜特征图文件夹
    EPOCHS = 30
    BATCH_SIZE = 512
    LEARNING_RATE = 0.005
    MARGIN = 2.0
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"当前使用的设备是: {DEVICE}")
    # --- 2. 准备数据 ---
    # 定义数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图片转换为Tensor，并归一化到[0,1]
        transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化到[-1,1]，适用于灰度图
    ])
    # 创建 Dataset 和 DataLoader
    val_dataset = SiameseIrisDataset(VAL_JSON_PATH, FEATURES_DIR, transform=transform) if "triple" not in VAL_JSON_PATH else SiameseIrisDatasetTriple(VAL_JSON_PATH, FEATURES_DIR, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    # --- 3. 初始化模型、损失函数和优化器 ---
    model = SiameseNetwork().to(DEVICE)
    state=model.load_state_dict(torch.load("best_siamese_model_triple.pth", map_location=DEVICE))
    # --- 验证 ---
    def eval_original():
        model.eval()


        pos_d, neg_d = [], []

        with torch.no_grad():
            for img1, img2, label in tqdm(val_loader):
                img1, img2, label = img1.cuda(), img2.cuda(), label.cuda()
                f1, f2 = model(img1),model(img2)
                d = F.pairwise_distance(f1, f2).cpu().numpy()

                label = label.cpu().numpy()

                pos_d.extend(d[label == 1])
                neg_d.extend(d[label == 0])
        numpy.savez_compressed("test_result_triple.npz", pos_d=pos_d, neg_d=neg_d)
        print("正样本平均距离:", sum(pos_d) / len(pos_d))
        print("负样本平均距离:", sum(neg_d) / len(neg_d))


    def eval():
        model.eval()

        pos_d, neg_d = [], []

        with torch.no_grad():
            for a, p, n in tqdm(val_loader):
                a, p, n = a.to(DEVICE), p.to(DEVICE), n.to(DEVICE)
                output_a, output_p, output_n = model(a), model(p), model(n)
                dp = F.pairwise_distance(output_a, output_p).cpu().numpy()
                dn= F.pairwise_distance(output_a, output_n).cpu().numpy()

                pos_d.extend(dp)
                neg_d.extend(dn)
        numpy.savez_compressed("test_result_triple_new.npz", pos_d=pos_d, neg_d=neg_d)
        print("正样本平均距离:", sum(pos_d) / len(pos_d))
        print("负样本平均距离:", sum(neg_d) / len(neg_d))
    eval()