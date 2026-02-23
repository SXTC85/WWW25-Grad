import numpy as np
import os

# 1. 确保 data 文件夹存在
os.makedirs('./data', exist_ok=True)

# 2. 模拟 DGraph-Fin 的规格 (缩小 1000 倍)
num_nodes = 1000       # 1000 个节点足够本地 CPU 运行了
feat_dim = 17          # ！！！核心：必须是 17 维，对标 DGraph-Fin

# 随机生成特征矩阵 [N, 17]
x = np.random.rand(num_nodes, feat_dim).astype(np.float32)

# 随机生成标签 [N]，只有 0 (正常) 和 1 (欺诈)
y = np.random.randint(0, 2, size=num_nodes)

# 随机生成边关系 [E, 2]
# 注意：DGraph-Fin 原始格式是 [边数, 2]，你的 dataProcess 会执行 .t() 转换
num_edges = 3000
edge_index = np.random.randint(0, num_nodes, size=(num_edges, 2))

# 3. 保存为你的 dataProcess.py 预期的文件名
np.savez('./data/dgraph-fin.npz', x=x, y=y, edge_index=edge_index)

print(f"成功运行！已在 ./data 目录下生成微型数据集。")
print(f"节点数: {num_nodes}, 特征维度: {feat_dim}")
