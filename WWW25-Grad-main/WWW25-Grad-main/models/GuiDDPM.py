import argparse
import inspect
import torch
import torch.utils.data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Batch

from .improved_diffusion import gaussian_diffusion as gd
from .improved_diffusion.respace import SpacedDiffusion, space_timesteps
from .improved_diffusion.unet import UNetModel
from .improved_diffusion import logger
from .improved_diffusion.resample import create_named_schedule_sampler
from .improved_diffusion.train_util import TrainLoop
from .improved_diffusion import dist_util

import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
import numpy as np
import os
from tqdm import tqdm

import sys

sys.path.append('../')
from utils.MyUtils import color_print


# 1. 训练数据集类
class DDPMTrainDataset(torch.utils.data.Dataset):
    def __init__(self, datalist):
        self.datalist = datalist
        self.length = len(datalist)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.datalist[idx], {}


# 2. 采样数据集类
class DDPMSampleDataset(torch.utils.data.Dataset):
    def __init__(self, datalist):
        self.datalist = datalist
        self.length = len(datalist)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_item = self.datalist[idx]
        data_dict = {
            'x': data_item.x,
            'edge_index': data_item.edge_index,
            'y': data_item.y
        }
        return data_item.x, data_dict


def create_model_and_diffusion(args, diffusion_steps):
    """创建 17 维输入的 GNN 扩散模型"""
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
        dist_util.setup_dist()

    def trainDataLoader(self):
        """核心重构：使用自定义 Batch 生成器，彻底避开 DataLoader 死锁"""

        def fast_batch_gen(groups, batch_size):
            import random
            while True:
                random.shuffle(groups)
                for i in range(0, len(groups), batch_size):
                    chunk = groups[i: i + batch_size]
                    if not chunk: continue
                    # 将多个子图打包成一个大 Batch 喂给显卡
                    yield (Batch.from_data_list(chunk), {})

        return fast_batch_gen(self.node_groups, self.global_args.GuiDDPM_train_diffusion_batch_size)

    def train(self, train_steps):
        logger.configure(dir='tmp/')
        self.model, self.diffusion = create_model_and_diffusion(
            self.global_args,
            self.global_args.GuiDDPM_train_diffusion_steps
        )
        self.model.to(self.device)
        color_print(f'GuiDDPM GNN Backbone Loaded Success')

        self.schedule_sampler = create_named_schedule_sampler("uniform", self.diffusion)
        # 使用升级后的生成器
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
        """核心重构：加固采样逻辑，防止维度报错和设备冲突"""
        logger.log(">>> 正在为检测阶段准备模型和子图采样...")
        self.model, self.diffusion = create_model_and_diffusion(
            self.global_args,
            self.global_args.GuiDDPM_sample_diffusion_steps
        )
        self.model.to(self.device)  # 确保模型在 GPU

        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            color_print(f">>> 权重加载成功: {self.model_path}")

        self.model.eval()
        self.sample_dataset = DDPMSampleDataset(self.node_groups)
        # 采样阶段必须 batch_size=1 以处理不同边数的子图
        self.sample_data_loader = PyGDataLoader(
            dataset=self.sample_dataset,
            batch_size=1,
            num_workers=0
        )

        logger.log(">>> 正在通过扩散恢复提取异常信号...")
        all_mse_scores = []

        for i, (batch_x, data_dict) in enumerate(tqdm(self.sample_data_loader)):
            # 剥离 batch 维度 [1, N, 17] -> [N, 17]
            x = batch_x.squeeze(0).to(self.device).float()
            edge_index = data_dict['edge_index'].squeeze(0).to(self.device)

            with torch.no_grad():
                samples = self.diffusion.p_sample_loop(
                    self.model,
                    x.shape,
                    clip_denoised=True,
                    model_kwargs={'edge_index': edge_index}
                )
                mse = torch.mean((x - samples) ** 2, dim=1)
                all_mse_scores.append(mse.cpu())

        # 保存关系字典
        syn_relation_dict = {
            'syn_relation_list': self.node_groups,
            'unselected_edge_index': self.edge_index_unselected,
            'graph_pyg_ssupgcl_new_x': self.graph_pyg_ssupgcl.new_x
        }
        os.makedirs(os.path.dirname(self.syn_relation_filename), exist_ok=True)
        torch.save(syn_relation_dict, self.syn_relation_filename)
        color_print(f'!!!!! 关系生成成功')

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        color_print(f'!!!!! GuiDDPM model saved')
