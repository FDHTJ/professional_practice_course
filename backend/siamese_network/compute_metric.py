import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import json
# print(len(json.load(open("train_triple.json",'r'))))
# print(len(json.load(open("val_triple.json",'r'))))
# print(len(json.load(open("test_triple.json",'r'))))

data=np.load("test_result.npz")
pos_d=data["pos_d"]
neg_d=data["neg_d"]
def plot_roc_curve():
    distances = np.concatenate([pos_d, neg_d])
    labels = np.concatenate([np.ones_like(pos_d), np.zeros_like(neg_d)])

    # 距离越小越相似，所以取 -distance
    fpr, tpr, _ = roc_curve(labels, -distances)
    roc_auc = auc(fpr, tpr)

    print("ROC AUC:", roc_auc)

    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
def compute_eer():
    """
    pos_d：正样本距离列表（anchor-positive）
    neg_d：负样本距离列表（anchor-negative）
    """

    # 构造标签：正样本 1，负样本 0
    y_true = np.array([1] * len(pos_d) + [0] * len(neg_d))
    # 距离越大越不像，所以取负号变成“越大越相似”
    y_score = -np.concatenate([pos_d, neg_d])

    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # EER: FPR = 1 - TPR
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fnr - fpr))

    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    threshold = thresholds[eer_index]

    return eer, threshold
def plot_hist():
    plt.figure(figsize=(8, 5))

    # 绘制直方图
    plt.hist(pos_d, bins=50, alpha=0.6, label="Positive Distance", density=True)
    plt.hist(neg_d, bins=50, alpha=0.6, label="Negative Distance", density=True)

    plt.xlabel("Distance")
    plt.ylabel("Density")
    plt.title("Distance Distribution (Positive vs Negative)")
    plt.legend()

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    plt.show()
plot_hist()
e,t=compute_eer()
print("err:",e)
print("threshold:",t)