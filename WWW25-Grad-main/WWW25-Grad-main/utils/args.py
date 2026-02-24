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
        # 【注意】：现在我们要进行异常检测，所以必须设为 False
        self.GuiDDPM_train_flag = False

        # 3. 引导与融合参数
        self.GuiDDPM_sample_guidance_scale = 15
        self.GuiDDPM_sample_with_guidance = True
        self.GuiDDPM_sample_diffusion_steps = 100
        self.WFusion_hid_dim = 256
        self.WFusion_order = 5
        self.WFusion_epochs = 250
        self.WFusion_gdc_syn_avg_degree = []
        self.WFusion_gdc_raw_avg_degree = []
        self.WFusion_relation_index = [0, 1]
        self.WFusion_use_WFusion = True

        # 4. SupGCL 参数 (为了本地能跑通，暂时保留)
        self.SupGCL_batch_size = 2
        self.SupGCL_num_train_part = 20
        self.SupGCL_epochs = 50
        self.SupGCL_train_flag = False # 本地跳过 GCL
        self.SupGCL_visualize_flag = False

        # 5. 【环境自适应】：根据是否有 GPU 自动切换性能参数
        if torch.cuda.is_available():
            # Kaggle GPU 正式参数
            self.nodes_per_subgraph = 256
            self.GuiDDPM_train_diffusion_batch_size = 64
            self.GuiDDPM_sample_diffusion_batch_size = 64
            self.GuiDDPM_train_diffusion_steps = 1000
            self.GuiDDPM_train_steps = 6000
            print(">>> 检测到 GPU：启用 Kaggle 正式训练参数模式")
        else:
            # 本地 CPU 调试参数
            self.nodes_per_subgraph = 32
            self.GuiDDPM_train_diffusion_batch_size = 2
            self.GuiDDPM_sample_diffusion_batch_size = 1
            self.GuiDDPM_train_diffusion_steps = 50
            self.GuiDDPM_train_steps = 10
            print(">>> 未检测到 GPU：启用本地 CPU 极速采样/检测模式")
