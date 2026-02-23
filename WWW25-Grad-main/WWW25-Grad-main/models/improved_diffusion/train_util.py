import copy
import functools
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler


# 为了在没有初始化分布式环境下安全获取 rank
def get_rank_safe():
    try:
        if dist.is_initialized():
            return dist.get_rank()
    except:
        pass
    return 0


INITIAL_LOG_LOSS_SCALE = 20.0


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
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0

        # 修复分布式获取 world_size 的报错
        try:
            if dist.is_initialized():
                self.global_batch = self.batch_size * dist.get_world_size()
            else:
                self.global_batch = self.batch_size
        except:
            self.global_batch = self.batch_size

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        # 针对 CPU 环境的 DDP 兼容处理
        if th.cuda.is_available() and dist.is_initialized():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if get_rank_safe() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )
        # 屏蔽同步报错
        try:
            dist_util.sync_params(self.model.parameters())
        except:
            pass


    def _anneal_lr(self):
        """
        学习率衰减逻辑：根据当前步数自动调整学习率
        """
        if not self.lr_anneal_steps:
            return
        # 计算完成比例
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        # 更新优化器中的学习率
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def run_loop(self):
        while (
                not self.lr_anneal_steps
                or self.step + self.resume_step < self.lr_anneal_steps
        ):
            # 核心：从 DataLoader 拿到的 batch 实际上是 PyG 的 Data 对象
            batch_data, cond = next(self.data)
            self.run_step(batch_data, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
            self.step += 1
            if get_rank_safe() == 0:
                print(f'Training Step: {self.step}', end='\r')

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    def forward_backward(self, batch, cond):
        zero_grad(self.model_params)

        # === 核心修改：针对图数据，取消微批次切片逻辑 ===
        # 直接获取 batch 中的全量数据，不再进行 i : i + microbatch 的循环
        micro = batch.x.to(dist_util.dev())
        edge_index = batch.edge_index.to(dist_util.dev())

        # 对齐条件变量
        micro_cond = {
            k: v.to(dist_util.dev())
            for k, v in cond.items()
        }

        # 采样时间步和权重
        t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

        # 计算损失
        # 此时 micro 是完整的全量节点，edge_index 也是完整的，不会再报越界错误
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.ddp_model,
            micro,
            t,
            edge_index=edge_index,
            model_kwargs=micro_cond,
        )

        if not self.use_ddp:
            losses = compute_losses()
        else:
            with self.ddp_model.no_sync():
                losses = compute_losses()

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )

        loss = (losses["loss"] * weights).mean()
        log_loss_dict(
            self.diffusion, t, {k: v * weights for k, v in losses.items()}
        )

        if self.use_fp16:
            loss_scale = 2 ** self.lg_loss_scale
            (loss * loss_scale).backward()
        else:
            loss.backward()

    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        for p in self.master_params:
            if p.grad is not None:
                sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        if get_rank_safe() == 0:
            def save_checkpoint(rate, params):
                state_dict = self._master_params_to_state_dict(params)
                logger.log(f"saving model {rate}...")
                filename = f"model{(self.step + self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

            save_checkpoint(0, self.master_params)

        try:
            dist.barrier()
        except:
            pass

    def _master_params_to_state_dict(self, master_params):
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        return [state_dict[name] for name, _ in self.model.named_parameters()]


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    return None


def parse_resume_step_from_filename(filename):
    split = filename.split("model")
    if len(split) < 2: return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except:
        return 0


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
