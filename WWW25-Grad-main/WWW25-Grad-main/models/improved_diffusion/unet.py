from abc import abstractmethod
from torch_geometric.nn import GATv2Conv
import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .nn import (
    SiLU,
    linear,
    zero_module,
    normalization,
    timestep_embedding,
)

class UNetModel(nn.Module):
    """
    针对图数据（DGraph-Fin）重新设计的扩散模型骨干网络
    放弃了图像用的卷积层，完全采用图注意力网络（GAT）
    """
    def __init__(
        self,
        in_channels,      # 17 (DGraph-Fin)
        model_channels,   # 128
        out_channels,     # 17 (重建目标)
        num_res_blocks=0, # 图模式下不使用原ResBlock
        attention_resolutions=None,
        dropout=0,
        channel_mult=(1,),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=4,      # 建议多头注意力设为4
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_heads = num_heads

        # 1. 时间步嵌入层 (保留扩散模型的核心)
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # 2. 核心图神经网络层：GATv2
        # 对标任务书：关系扩散机制。通过 edge_index 显式捕捉邻居特征
        self.gat_layer = GATv2Conv(
            in_channels=in_channels,
            out_channels=model_channels,
            heads=num_heads,
            dropout=dropout,
            concat=True
        )

        # 3. 维度投影（将多头 128*4 映射回 128）
        self.feature_projection = linear(model_channels * num_heads, model_channels)

        # 4. 时间信息融合层：将时间步信息注入节点特征
        self.time_embed_to_feature = nn.Sequential(
            SiLU(),
            linear(time_embed_dim, model_channels)
        )

        # 5. 输出层：重建原始特征，用于计算异常得分（MSE）
        self.out_layer = nn.Sequential(
            normalization(model_channels),
            SiLU(),
            zero_module(linear(model_channels, out_channels))
        )

        # 占位符：为了兼容一些旧代码的调用逻辑，防止报错
        self.input_blocks = nn.ModuleList([])
        self.middle_block = nn.Identity()
        self.output_blocks = nn.ModuleList([])

    @property
    def inner_dtype(self):
        return next(self.parameters()).dtype

    def forward(self, x, timesteps, edge_index=None, y=None):
        """
        核心数据流向：
        :param x: [N, 17] 节点特征
        :param edge_index: [2, E] 图连接关系
        :param timesteps: 扩散步数
        """
        # A. 计算扩散的时间嵌入
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        # B. 类型转换（保持和模型权重一致）
        h = x.type(self.inner_dtype)

        # C. 【重要：关系扩散】执行图注意力卷积
        # 即使欺诈者做了特征伪装，这里通过 edge_index 强行引入邻居信息，
        # 会在注意力权重上产生异常，导致重建结果偏离，从而识别欺诈。
        h = self.gat_layer(h, edge_index)
        h = self.feature_projection(h)
        h = F.silu(h)

        # D. 融入时间步信息
        h = h + self.time_embed_to_feature(emb)

        # E. 最后输出重建后的 17 维特征
        return self.out_layer(h)

    def get_feature_vectors(self, x, timesteps, edge_index=None, y=None):
        # 简易版特征提取逻辑
        return {"middle": self.forward(x, timesteps, edge_index, y)}

