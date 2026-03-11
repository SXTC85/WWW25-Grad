import torch


class argVar:
    def __init__(self):
        # 1. 基础数据集设置 (对标 DGraph-Fin)
        self.dataset = 'dgraph-fin'
        self.num_classes = 2
        self.in_channels = 17
        self.out_channels = 17
        self.train_ratio = 0.4
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 2. **核心开关：训练 or 采样**
        # 默认设为 False，因为我们现在已经练好了模型，主要进行检测
        self.GuiDDPM_train_flag = False

        # 3. 扩散模型核心参数
        self.GuiDDPM_train_steps = 6000  # 固化为 6000 步
        self.GuiDDPM_sample_diffusion_steps = 100  # 加速采样设为 100 步
        self.GuiDDPM_train_diffusion_batch_size = 16  # 最佳 Batch Size
        self.nodes_per_subgraph = 256  # 最佳子图大小

        # 4. 引导与融合参数 (保留逻辑完整性)
        self.GuiDDPM_sample_guidance_scale = 15
        self.GuiDDPM_sample_with_guidance = True
        self.WFusion_hid_dim = 256
        self.WFusion_order = 5
        self.WFusion_epochs = 250
        self.WFusion_relation_index = [0, 1]
        self.WFusion_use_WFusion = True

        # 5. 其他模块开关
        self.SupGCL_train_flag = False
        self.SupGCL_epochs = 50

        # 打印状态
        if torch.cuda.is_available():
            print(f">>> [System] 已就绪：使用 GPU 模式，加载 {self.GuiDDPM_train_steps} 步模型配置")
        else:
            print(f">>> [System] 已就绪：使用 CPU 模式（评估会较慢），加载 {self.GuiDDPM_train_steps} 步模型配置")
