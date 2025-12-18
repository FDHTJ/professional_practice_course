import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm

# --- 1. 配置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_DIR = 'datasets/labeled_images'
MASK_DIR = 'datasets/processed_masks'
BATCH_SIZE = 4  # 如果你的GPU显存不够，可以调小
EPOCHS = 25  # 训练轮数
LR = 0.001  # 学习率
# --- 5. 模型、损失函数、优化器 ---
model = smp.Unet(
    encoder_name="mobilenet_v2", # <--- 替换为 MobileNetV2
    encoder_weights=None,   # 同样使用预训练权重，加速收敛
    in_channels=3,
    classes=3,
)

weights=torch.load("pretrained_mobilenet_v2_segmentation_model.pth",map_location=DEVICE)
model.load_state_dict(weights)
model.to(DEVICE)
# --- 2. 自定义数据集 ---
class PigeonEyeDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            # 将掩码的类型转为LongTensor，用于损失计算
            mask = mask.long()

        return image, mask


# --- 3. 数据增强 ---
# 训练集使用较强的数据增强
train_transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# 验证集只做必要的尺寸和归一化处理
val_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# --- 4. 准备数据加载器 ---
# 假设你的原始图片和处理后的掩码文件名可以对应
# 例如 original_images/task_1.jpg 对应 processed_masks/mask_1.png
# 你需要确保这个对应关系是正确的
all_images = sorted([os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)])
all_masks = sorted([os.path.join(MASK_DIR, f) for f in os.listdir(MASK_DIR)])

# 划分训练集和验证集 (80% 训练, 20% 验证)
train_images, val_images, train_masks, val_masks = train_test_split(
    all_images, all_masks, test_size=0.1, random_state=42
)

train_dataset = PigeonEyeDataset(train_images, train_masks, transform=train_transform)
val_dataset = PigeonEyeDataset(val_images, val_masks, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)



# DiceLoss在分割任务中表现很好
loss_fn = smp.losses.DiceLoss(mode='multiclass')
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
# IoU (Intersection over Union) 是分割任务常用指标
metrics = [smp.metrics.iou_score]

# --- 6. 训练循环 ---
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    model.train()
    train_loss = 0
    for images, masks in tqdm(train_loader):
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Train Loss: {train_loss / len(train_loader):.4f}")

    # --- 验证 ---
    model.eval()
    val_iou = 0
    with torch.no_grad():
        for images, masks in tqdm(val_loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            pred_masks = torch.argmax(outputs, dim=1)
            tp, fp, fn, tn=smp.metrics.get_stats(
            pred_masks,
            masks,
            mode='multiclass',
            threshold=None,
            num_classes=3,
            # ignore_index=0  # (可选，但推荐) 忽略背景类别(索引为0)的计算，我们只关心瞳孔和虹膜的IoU
        )
            # 计算IoU分数
            iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='macro')
            val_iou += iou.item()

    print(f"Validation IoU: {val_iou / len(val_loader):.4f}")

# --- 7. 保存模型 ---
torch.save(model.state_dict(), '../../backend/main/segmentation/pigeon_eye_segmentation_model.pth')
print("模型已保存!")
