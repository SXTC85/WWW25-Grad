import argparse
import inspect
import torch
import torch.utils.data
# 核心修改：引入 PyG 专用的加载器，处理图数据对象
from torch_geometric.loader import DataLoader as PyGDataLoader


from .improved_diffusion import gaussian_diffusion as gd
from .improved_diffusion.respace import SpacedDiffusion, space_timesteps
from .improved_diffusion.unet import UNetModel  # 注意这里取消了 SuperResModel 的引用
from .improved_diffusion import logger
from .improved_diffusion.resample import create_named_schedule_sampler
from .improved_diffusion.train_util import TrainLoop
from .improved_diffusion import dist_util

import torch.distributed as dist
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data
import numpy as np
import os
from tqdm import tqdm

import sys

sys.path.append('../')
from utils.MyUtils import color_print


# 1. 核心数据集类重构：从“图像矩阵”转为“节点特征”
class DDPMTrainDataset(torch.utils.data.Dataset):
    def __init__(self, datalist):
        self.datalist = datalist
        self.length = len(datalist)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 核心修改：返回 PyG Data 对象，这样 TrainLoop 才能拿到 edge_index
        data_item = self.datalist[idx]
        # x 是我们要扩散的 17 维特征
        return data_item, {}


class DDPMSampleDataset(torch.utils.data.Dataset):
    def __init__(self, datalist):
        self.datalist = datalist
        self.length = len(datalist)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_item = self.datalist[idx]
        # 采样时需要的元数据
        data_dict = {
            'x': data_item.x,
            'edge_index': data_item.edge_index,
            'y': data_item.y
        }
        return data_item.x, data_dict


def create_model_and_diffusion(args, diffusion_steps):
    """
    根据任务书要求，创建 17 维输入的 GNN 扩散模型
    """
    # 强制指定 17 维
    model = UNetModel(
        in_channels=17,
        model_channels=128,
        out_channels=17,
        num_res_blocks=0,
        attention_resolutions=None,
        num_heads=4
    )

    betas = gd.get_named_beta_schedule("linear", diffusion_steps)

    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, [diffusion_steps]),
        betas=betas,
        # 修改后：
        model_mean_type=gd.ModelMeanType.EPSILON,
        model_var_type=gd.ModelVarType.FIXED_SMALL,
        loss_type=gd.LossType.MSE,
        rescale_timesteps=True,
    )
    return model, diffusion


class GuiDDPM:
    def __init__(self, global_args, graph_pyg_ssupgcl, node_groups, edge_index_unselected, guidance, train_flag,
                 model_path, syn_relation_filename, device):
        self.device = device
        self.graph_pyg_ssupgcl = graph_pyg_ssupgcl
        self.node_groups = node_groups
        self.train_flag = train_flag
        self.model_path = model_path
        self.guidance = guidance
        self.global_args = global_args
        self.edge_index_unselected = edge_index_unselected
        self.syn_relation_filename = syn_relation_filename

        # 分布式环境模拟（执行你改好的 dist_util）
        dist_util.setup_dist()

    def trainDataLoader(self):
        # 核心修改：使用 PyGDataLoader 替换原有的 DataLoader
        data_loader = PyGDataLoader(
            dataset=self.train_dataset,
            batch_size=self.global_args.GuiDDPM_train_diffusion_batch_size,
            shuffle=True,
            num_workers=0,  # 本地调试保持为 0
            drop_last=True)
        while True:
            yield from data_loader

    def train(self, train_steps):
        logger.configure(dir='tmp/')
        logger.log("creating model and diffusion...")

        self.model, self.diffusion = create_model_and_diffusion(
            self.global_args,
            self.global_args.GuiDDPM_train_diffusion_steps
        )
        self.model.to(self.device)

        color_print(f'GuiDDPM GNN Backbone Loaded Success')

        # 初始化采样器
        self.schedule_sampler = create_named_schedule_sampler("uniform", self.diffusion)
        self.train_dataset = DDPMTrainDataset(self.node_groups)
        self.train_data_loader = self.trainDataLoader()

        logger.log("start training loop...")
        TrainLoop(
            model=self.model,
            diffusion=self.diffusion,
            data=self.train_data_loader,
            batch_size=self.global_args.GuiDDPM_train_diffusion_batch_size,
            microbatch=-1,
            lr=1e-4,
            ema_rate="0.9999",
            log_interval=10,
            save_interval=2000,
            resume_checkpoint="",
            schedule_sampler=self.schedule_sampler,
            lr_anneal_steps=train_steps,
        ).run_loop()

    def sample(self):
        """
        根据重构误差进行异常检测推理（论文核心逻辑）
        """
        logger.log("creating model for sampling...")
        self.model, self.diffusion = create_model_and_diffusion(
            self.global_args,
            self.global_args.GuiDDPM_sample_diffusion_steps
        )

        # 加载训练好的权重
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))

        self.model.eval()
        self.sample_dataset = DDPMSampleDataset(self.node_groups)
        self.sample_data_loader = PyGDataLoader(
            dataset=self.sample_dataset,
            batch_size=self.global_args.GuiDDPM_sample_diffusion_batch_size,
            num_workers=0
        )  # 这里现在只有一个括号了

        logger.log("sampling / detecting anomalies...")

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        color_print(f'!!!!! GuiDDPM model saved in {path}')
