import copy
import functools
import os
import blobfile as bf
import numpy as np
import torch as th
from . import dist_util, logger
from .fp16_util import zero_grad
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler


class TrainLoop:
    def __init__(
            self,
            *,
            model,
            diffusion,
            data,
            batch_size,
            microbatch,
            lr,
            ema_rate,
            log_interval,
            save_interval,
            resume_checkpoint,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            schedule_sampler=None,
            weight_decay=0.0,
            lr_anneal_steps=0,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.model_params = list(self.model.parameters())
        self.opt = th.optim.AdamW(self.model_params, lr=self.lr, weight_decay=self.weight_decay)

        # 初始化 EMA 参数
        self.ema_params = [
            copy.deepcopy(self.model_params) for _ in range(len(self.ema_rate))
        ]

        if self.resume_checkpoint:
            self._load_and_sync_parameters()

    def _load_and_sync_parameters(self):
        print(f"loading model from checkpoint: {self.resume_checkpoint}...")
        self.model.load_state_dict(
            dist_util.load_state_dict(self.resume_checkpoint, map_location=dist_util.dev())
        )

    def run_loop(self):
        print(f">>> [Start] 核心训练启动 | Batch Size: {self.batch_size}", flush=True)
        while (
                not self.lr_anneal_steps
                or self.step < self.lr_anneal_steps
        ):
            # 1. 抓取 Batch 数据
            batch_data, cond = next(self.data)

            # 2. 执行训练步
            self.run_step(batch_data, cond)

            # 3. 定期保存
            if self.step > 0 and self.step % self.save_interval == 0:
                self.save()

            self.step += 1

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.model_params, rate=rate)

        # 每 10 步打印一次，确保换行刷新
        if self.step % 10 == 0:
            # 从 logger 获取最新均值（如果有）
            print(f'🚀 Step: {self.step} | 训练进行中...', flush=True)

    def forward_backward(self, batch, cond):
        zero_grad(self.model_params)

        # 把图数据搬到 GPU
        micro = batch.x.to(dist_util.dev())
        edge_index = batch.edge_index.to(dist_util.dev())

        # 核心修复：使用正确的采样函数获取时间步 t
        t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

        # 计算损失：确保 edge_index 透传给 GAT
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.model,
            micro,
            t,
            model_kwargs={'edge_index': edge_index},
        )

        losses = compute_losses()

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )

        loss = (losses["loss"] * weights).mean()

        # 实时打印当前 Loss
        if self.step % 10 == 0:
            print(f'🚀 Step: {self.step} | Loss: {loss.item():.4f}', flush=True)

        loss.backward()

    def save(self):
        filename = f"model{(self.step):06d}.pt"
        print(f">>> 正在保存模型至: {filename}")
        with bf.BlobFile(os.path.join(logger.get_dir(), filename), "wb") as f:
            th.save(self.model.state_dict(), f)


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
