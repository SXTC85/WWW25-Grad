"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
try:
    from mpi4py import MPI
except ImportError:
    # 针对 Windows CPU 环境的伪装逻辑
    # 只要让程序认为当前只有 1 个进程在跑即可
    # 找到 class MockMPI 并替换成这段：
    class MockMPI:
        def __init__(self):
            self.COMM_WORLD = self
            self.rank = 0
            self.size = 1

        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

        def barrier(self):
            pass

        def bcast(self, val, root=0):
            return val

import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    针对 Windows CPU 调试环境的终极简化：
    彻底跳过 init_process_group，防止 libuv 报错。
    """
    import os
    # 设置基础环境变量，防止其他地方调用时报错
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    print("已绕过 Distributed 初始化，进入单机调试模式。")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda:{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    if MPI.COMM_WORLD.Get_rank() == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
    else:
        data = None
    data = MPI.COMM_WORLD.bcast(data)
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
