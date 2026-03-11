"""
Helpers for distributed training.
Modified for Single-Node / Kaggle / Windows compatibility.
"""

import io
import os
import torch as th
import blobfile as bf

# ==========================================
# 核心修复：伪装 MPI 环境
# ==========================================
class MockMPI:
    def __init__(self):
        self.COMM_WORLD = self
    def Get_rank(self):
        return 0
    def Get_size(self):
        return 1
    def barrier(self):
        pass
    def bcast(self, val, root=0):
        return val

# 实例化 MockMPI，防止代码中引用 MPI 时报错
MPI = MockMPI()

def setup_dist():
    """
    单机模式的环境初始化：
    不再调用 th.distributed.init_process_group，彻底避免死锁。
    """
    # 设置基础环境变量，防止其他库读取失败
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    print(">>> [dist_util] 已进入单机极速模式（已绕过分布式同步）")

def dev():
    """
    智能选择设备：
    如果显卡可用，始终返回 cuda:0，否则返回 cpu。
    """
    if th.cuda.is_available():
        return th.device("cuda:0")
    return th.device("cpu")

def load_state_dict(path, **kwargs):
    """
    加载模型权重：
    直接通过磁盘读取，不再进行多节点广播。
    """
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)

def sync_params(params):
    """
    单机模式下无需同步参数，设为空操作。
    """
    pass

def all_gather(tensors, **kwargs):
    """
    单机模式下直接返回原张量列表。
    """
    return tensors
