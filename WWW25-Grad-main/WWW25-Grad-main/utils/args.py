import torch

class argVar:
    def __init__(self):
        # 修改 1：数据集切换
        self.dataset = 'dgraph-fin'

        # 修改 2：类别数。DGraph-Fin 主要是 0 (正常) 和 1 (欺诈)
        # 任务书要求：识别欺诈实体，所以这里设为 2
        self.num_classes = 2
        # 添加这一行，确保模型知道输入是 DGraph-Fin 的 17 维金融特征
        self.in_channels = 17
        # 同时定义输出维度，异常检测的重建目标也是 17 维
        self.out_channels = 17

        # 修改 3：划分比例。金融数据异常稀疏，建议保持 0.4 或根据任务书微调
        self.train_ratio = 0.4

        # 修改 4：子图节点数（非常关键！）
        # 原项目设为 32（针对短序列推荐）。DGraph-Fin 是大图。
        # 本地调试建议设为 128 或 256；上 Kaggle 训练建议设为 1024。
        self.nodes_per_subgraph = 128

        # 自动识别 CPU 或 GPU（你本地没 GPU，这里会自动选 CPU）
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.SupGCL_batch_size=2
        self.SupGCL_num_train_part=20
        self.SupGCL_epochs=50
        self.SupGCL_train_flag=True
        self.SupGCL_visualize_flag=False ### visualization of SupGCL

        self.GuiDDPM_train_flag = True  # 【修改点 2】
        self.GuiDDPM_train_steps = 6000
        self.GuiDDPM_train_diffusion_steps = 1000
        self.GuiDDPM_train_diffusion_batch_size = 20
        self.GuiDDPM_sample_diffusion_batch_size = 256
        self.GuiDDPM_sample_guidance_scale = 15  # 【修改点 2】
        self.GuiDDPM_sample_with_guidance = True
        self.GuiDDPM_sample_diffusion_steps = 100
        
        self.WFusion_hid_dim=256
        self.WFusion_order=5
        self.WFusion_epochs=250
        self.WFusion_gdc_syn_avg_degree=[]
        self.WFusion_gdc_raw_avg_degree=[]
        if self.dataset == 'amazon':
            self.WFusion_relation_index = [0, ]
        elif self.dataset == 'yelp':
            self.WFusion_relation_index = [0, 1, ]
        elif self.dataset == 'dgraph-fin':  # 【修改点 3】
            self.WFusion_relation_index = [0, 1]

        self.WFusion_use_WFusion = True
        # utils/args.py

        # 在 __init__ 的最后部分
        if torch.cuda.is_available():
            # === 这里是 Kaggle GPU 正式参数 ===
            self.nodes_per_subgraph = 256
            self.GuiDDPM_train_diffusion_batch_size = 64
            self.GuiDDPM_train_diffusion_steps = 1000
            self.GuiDDPM_train_steps = 6000
            print(">>> 检测到 GPU：启用 Kaggle 正式训练参数模式")
        else:
            # === 这里是本地 CPU 调试参数 ===
            self.nodes_per_subgraph = 32
            self.GuiDDPM_train_diffusion_batch_size = 2
            self.GuiDDPM_train_diffusion_steps = 50
            self.GuiDDPM_train_steps = 10
            print(">>> 未检测到 GPU：启用本地 CPU 极速调试模式")
