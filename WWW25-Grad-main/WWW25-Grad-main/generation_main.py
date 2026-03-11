from utils.MyUtils import color_print
from utils.args import argVar
from utils.dataProcess import loadDataset, nodeSelect, nodeSample, sampleCheck
# from models.SupGCL import SupGCL # 已旁路
from models.GuiDDPM import GuiDDPM

import torch
import os
import argparse


def GuiDDPM_module(args, graph_pyg_supgcl, node_groups, edge_index_unselected, guidance):
    # 自动生成权重文件名
    GuiDDPM_para_filename = f"./ModelPara/GuiDDPMPara/GuiDDPM_{args.dataset}_{args.GuiDDPM_train_steps}steps_subgraphsize_{args.nodes_per_subgraph}.pt"

    # 自动生成采样结果文件名
    if args.GuiDDPM_sample_with_guidance:
        syn_relation_filename = f"./Generation/SynRelation_{args.dataset}_{args.GuiDDPM_sample_diffusion_steps}Samplesteps_{args.GuiDDPM_train_steps}Trainsteps_subgraphsize_{args.nodes_per_subgraph}_guided.pt"
    else:
        syn_relation_filename = f"./Generation/SynRelation_{args.dataset}_{args.GuiDDPM_sample_diffusion_steps}Samplesteps_{args.GuiDDPM_train_steps}Trainsteps_subgraphsize_{args.nodes_per_subgraph}_unguided.pt"

    model_DDPM = GuiDDPM(global_args=args,
                         node_groups=node_groups,
                         graph_pyg_ssupgcl=graph_pyg_supgcl,
                         edge_index_unselected=edge_index_unselected,
                         guidance=guidance,
                         train_flag=args.GuiDDPM_train_flag,
                         model_path=GuiDDPM_para_filename,
                         syn_relation_filename=syn_relation_filename,
                         device=args.device)

    if args.GuiDDPM_train_flag:
        # 训练模式
        if os.path.exists(GuiDDPM_para_filename):
            color_print(f'!!!!! GuiDDPM Parameter is already exists at {GuiDDPM_para_filename}')
            # 如果你想覆盖训练，可以注释掉下面这行，直接 model_DDPM.train()
            # model_DDPM.train(train_steps=args.GuiDDPM_train_steps)
        else:
            model_DDPM.train(train_steps=args.GuiDDPM_train_steps)
            model_DDPM.save_model(GuiDDPM_para_filename)
    else:
        # 采样检测模式
        model_DDPM.sample()


def sampleAnalysis(node_groups, nodes_per_subgraph):
    """分析采样后的子图统计信息"""
    cnt = 0
    num_edge = 0
    wrong_num = 0
    for g in node_groups:
        if g.x.shape[0] != nodes_per_subgraph:
            wrong_num += 1
        cnt += 1
        num_edge += g.edge_index.shape[1]
    color_print(f'>>> 采样分析完成: 总子图数={cnt}, 总边数={num_edge}, 异常图数={wrong_num}')


def main():
    args = argVar()
    print(f'>>> 当前运行设备: {args.device}')

    # 1. 加载数据
    _, graph_pyg, train_mask, val_mask, test_mask = loadDataset(dataset=args.dataset, train_ratio=args.train_ratio)
    print(
        f'>>> 标签分布 (Train/Val/Test): {graph_pyg.y[train_mask].sum()} / {graph_pyg.y[val_mask].sum()} / {graph_pyg.y[test_mask].sum()}')

    # 2. 节点选择 (根据拓扑结构选择核心训练节点)
    graph_pyg_selected, edge_index_unselected = nodeSelect(graph_pyg=graph_pyg,
                                                           nodes_per_subgraph=args.nodes_per_subgraph)

    # 3. 旁路 SupGCL，直接使用原始特征进入扩散阶段
    graph_pyg_supgcl = graph_pyg

    # 4. 【核心修改】全图视野采样逻辑
    # 为了填满 Batch Size 16 并且增加模型泛化能力，我们循环采样 100 次
    node_groups = []
    print(
        f'>>> 正在执行全图多样性采样 (目标: 获取足够子图以支持 BatchSize={args.GuiDDPM_train_diffusion_batch_size})...')
    for _ in range(100):
        node_groups += nodeSample(graph_pyg=graph_pyg_supgcl, nodes_per_subgraph=args.nodes_per_subgraph)

    # 检查采样结果
    sampleAnalysis(node_groups, args.nodes_per_subgraph)

    # 5. 启动 GuiDDPM 模块 (训练或采样)
    GuiDDPM_module(args=args,
                   graph_pyg_supgcl=graph_pyg_supgcl,
                   node_groups=node_groups,
                   edge_index_unselected=edge_index_unselected,
                   guidance=None)


if __name__ == '__main__':
    # 开启 CUDA 异常追踪
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    main()
