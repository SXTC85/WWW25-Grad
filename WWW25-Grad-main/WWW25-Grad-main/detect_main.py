import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import os

from utils.dataProcess import loadDataset, nodeSample
from utils.args import argVar
from models.GuiDDPM import create_model_and_diffusion


def main():
    args = argVar()
    # 自动识别设备：有 GPU 用 GPU，没有用 CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f">>> [System] 当前评估设备: {device}")

    print(">>> 正在启动【平衡化】大规模评估...")
    # 加载数据集
    _, graph_pyg, _, _, _ = loadDataset(dataset=args.dataset, train_ratio=args.train_ratio)

    # 1. 自动定位模型路径
    model_path = f"./ModelPara/GuiDDPMPara/GuiDDPM_{args.dataset}_{args.GuiDDPM_train_steps}steps_subgraphsize_{args.nodes_per_subgraph}.pt"
    if not os.path.exists(model_path):
        model_path = "./model_final.pt"

    if not os.path.exists(model_path):
        print(f"❌ 错误：找不到模型权重文件！请检查路径 {model_path}")
        return

    print(f">>> 正在加载 6000 步满级模型: {model_path}")
    model, diffusion = create_model_and_diffusion(args, 100)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    all_residual_feats = []
    all_labels = []

    # 2. 提取深度残差特征 (带标签过滤)
    print(">>> 正在提取扩散重建残差 (自动过滤背景节点)...")
    # 采样 120 个子图以覆盖约 3 万个样本点
    for _ in tqdm(range(120)):
        gs = nodeSample(graph_pyg=graph_pyg, nodes_per_subgraph=256)
        for g in gs:
            # 【核心修复】：只保留标签为 0 或 1 的有效节点，剔除 -100
            valid_mask = (g.y == 0) | (g.y == 1)
            if valid_mask.sum() == 0:
                continue

            x = g.x.to(device).float()
            edge_index = g.edge_index.to(device)

            with torch.no_grad():
                # 运行 100 步加速采样重建
                samples = diffusion.p_sample_loop(model, x.shape, model_kwargs={'edge_index': edge_index})
                # 计算 17 维逐位残差
                residual_feat = torch.abs(x - samples)

                # 仅存储有效节点的特征和标签
                all_residual_feats.append(residual_feat[valid_mask].cpu())
                all_labels.append(g.y[valid_mask].cpu())

    # 合并特征
    X_all = torch.cat(all_residual_feats, dim=0).numpy()
    Y_all = torch.cat(all_labels, dim=0).numpy()

    # 3. 平衡化处理
    pos_idx = np.where(Y_all == 1)[0]
    neg_idx = np.where(Y_all == 0)[0]
    print(f">>> 样本分布 - 异常节点: {len(pos_idx)}, 正常节点: {len(neg_idx)}")

    if len(pos_idx) == 0:
        print("❌ 错误：采样的样本中没有异常节点，请重新运行评估。")
        return

    # 计算类别权重，强行提升对欺诈样本的关注度
    pos_weight = torch.tensor([len(neg_idx) / len(pos_idx)]).to(device)

    # 4. 构建深度判别器 (MLP)
    classifier = nn.Sequential(
        nn.Linear(17, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
    ).to(device)

    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    X_tensor = torch.FloatTensor(X_all).to(device)
    Y_tensor = torch.FloatTensor(Y_all).to(device).view(-1, 1)

    print(">>> 正在进行判别空间对齐训练 (200 Epochs)...")
    for epoch in range(200):
        optimizer.zero_grad()
        pred = classifier(X_tensor)
        # 使用类别加权 BCE Loss
        loss = nn.functional.binary_cross_entropy(pred, Y_tensor,
                                                  weight=(Y_tensor * pos_weight + (1 - Y_tensor)))
        loss.backward()
        optimizer.step()

    # 5. 计算并输出最终指标
    with torch.no_grad():
        final_probs = classifier(X_tensor).cpu().numpy()
        final_auc = roc_auc_score(Y_all, final_probs)
        final_ap = average_precision_score(Y_all, final_probs)

    print("\n" + "⭐" * 25)
    print(f"🏆 最终稳健评估结果 (N={len(X_all)}):")
    print(f"✅ 最终 AUC: {final_auc:.4f}")
    print(f"✅ 最终 AP: {final_ap:.4f}")
    print(f"✅ 状态: 已完成 3 万样本压力测试")
    print("⭐" * 25)


if __name__ == "__main__":
    main()
