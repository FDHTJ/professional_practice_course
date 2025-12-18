import os
import json
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

    # def forward_one(self, x):
    #     # 用于推理时，提取单张图片的特征
    #     return self.base_net(x)

    def forward(self, image):
        # 用于训练时，提取一对图片的特征
        output = self.base_net(image)
        return output


# ==============================================================================
# 步骤 3: 定义对比损失函数
# ==============================================================================
#


# ==============================================================================
# 步骤 4: 完整的训练和验证流程
# ==============================================================================
if __name__ == '__main__':

    # --- 1. 配置参数 ---
    TRAIN_JSON_PATH = "train_triple.json"
    VAL_JSON_PATH = "val_triple.json"
    FEATURES_DIR = "../datasets/xinge_jpg/iris_fetures"  # 您的虹膜特征图文件夹

    EPOCHS = 100
    BATCH_SIZE = 256
    LEARNING_RATE = 0.0005
    MARGIN = 1.0

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"当前使用的设备是: {DEVICE}")

    # --- 2. 准备数据 ---
    # 定义数据预处理
    transform =transforms.Compose([
        transforms.ToTensor(),  # 将图片转换为Tensor，并归一化到[0,1]
        transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化到[-1,1]，适用于灰度图
    ])

    # 创建 Dataset 和 DataLoader
    train_dataset = SiameseIrisDataset(TRAIN_JSON_PATH, FEATURES_DIR, transform=transform)
    val_dataset = SiameseIrisDataset(VAL_JSON_PATH, FEATURES_DIR, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # --- 3. 初始化模型、损失函数和优化器 ---
    model = SiameseNetwork().to(DEVICE)
    triple_loss=torch.nn.TripletMarginLoss(margin=MARGIN)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 4. 训练和验证循环 ---
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        # --- 训练 ---
        model.train()
        train_loss = 0.0
        # 使用 tqdm 显示进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [训练]")
        for a,p,n in pbar:
            a,p,n = a.to(DEVICE), p.to(DEVICE), n.to(DEVICE)

            optimizer.zero_grad()
            output_a, output_p,output_n = model(a),model(p),model(n)
            loss = triple_loss(output_a,output_p,output_n)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"Loss": loss.item()})

        avg_train_loss = train_loss / len(train_loader)

        # --- 验证 ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [验证]")
            for a,p,n in vbar:
                a, p, n = a.to(DEVICE), p.to(DEVICE), n.to(DEVICE)
                output_a, output_p, output_n = model(a), model(p), model(n)
                loss = triple_loss(output_a, output_p, output_n)
                val_loss += loss.item()
                vbar.set_postfix({"Loss": loss.item()})

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}/{EPOCHS} -> 训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}")

        # --- 保存最佳模型 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_siamese_model_triple.pth")
            print(f"*** 模型已更新！新的最佳验证损失: {best_val_loss:.4f} ***")

    print("训练完成！最佳模型已保存为 'best_siamese_model_triple.pth'")