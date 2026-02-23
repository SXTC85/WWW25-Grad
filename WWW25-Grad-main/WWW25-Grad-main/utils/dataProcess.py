from .MyUtils import color_print  # 去掉了引发错误的 pyg_data_to_dgl_graph

import torch
import numpy as np

# 注释掉报错的 DGL 库
# import dgl
# from dgl.data import FraudAmazonDataset, FraudYelpDataset

import torch_geometric
from torch_geometric.transforms import GDC
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.utils import to_dense_adj, dense_to_sparse

from sklearn.model_selection import train_test_split
from tqdm import tqdm


def splitDataset(index, label, train_ratio):
    idx_train, idx_rest, y_train, y_rest = train_test_split(index, label[index], stratify=label[index],
                                                            train_size=train_ratio,
                                                            random_state=2, shuffle=True)
    idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                            test_size=0.5,
                                                            random_state=2, shuffle=True)
    train_mask = torch.zeros([len(label)]).bool()
    val_mask = torch.zeros([len(label)]).bool()
    test_mask = torch.zeros([len(label)]).bool()
    train_mask[idx_train] = 1
    val_mask[idx_valid] = 1
    test_mask[idx_test] = 1

    return train_mask, val_mask, test_mask


def nodeSelect(graph_pyg, nodes_per_subgraph):
    num = graph_pyg.x.shape[0]
    num_selected = num - num % nodes_per_subgraph

    if num_selected:
        feat_selected = graph_pyg.x[:num_selected]
        label_selected = graph_pyg.y[:num_selected]
        train_mask_selected = graph_pyg.train_mask[:num_selected]
        test_mask_selected = graph_pyg.test_mask[:num_selected]
        val_mask_selected = graph_pyg.val_mask[:num_selected]

        edge_index_selected_mask = (graph_pyg.edge_index[0] < num_selected) & (graph_pyg.edge_index[1] < num_selected)
        edge_index_unselected = graph_pyg.edge_index[:, ~edge_index_selected_mask]
        edge_index_selected = graph_pyg.edge_index[:, edge_index_selected_mask]

        graph_pyg_selected = Data(x=feat_selected,
                                  edge_index=edge_index_selected,
                                  y=label_selected,
                                  train_mask=train_mask_selected,
                                  val_mask=val_mask_selected,
                                  test_mask=test_mask_selected)

        print(f'num_seleced: {num_selected}/{num}')

    return graph_pyg_selected, edge_index_unselected


def nodeSample(graph_pyg, nodes_per_subgraph):
    color_print(f'!!!!! Start node sampling')
    num = graph_pyg.x.shape[0]
    num_selected = num - num % nodes_per_subgraph

    subgraphs = []

    for i in tqdm(range(num_selected // nodes_per_subgraph)):
        start_idx = i * nodes_per_subgraph

        subset = torch.arange(start_idx, start_idx + nodes_per_subgraph)

        sub_edge_index = torch_geometric.utils.subgraph(subset, graph_pyg.edge_index, num_nodes=graph_pyg.num_nodes)[
                             0] - i * nodes_per_subgraph
        sub_x = graph_pyg.x[start_idx:start_idx + nodes_per_subgraph]
        sub_new_x = graph_pyg.new_x[start_idx:start_idx + nodes_per_subgraph] if hasattr(graph_pyg, 'new_x') else sub_x
        sub_y = graph_pyg.y[start_idx:start_idx + nodes_per_subgraph]
        sub_adj = to_dense_adj(sub_edge_index, max_num_nodes=nodes_per_subgraph)[0]
        sub_train_mask = graph_pyg.train_mask[start_idx:start_idx + nodes_per_subgraph]
        sub_val_mask = graph_pyg.val_mask[start_idx:start_idx + nodes_per_subgraph]
        sub_test_mask = graph_pyg.test_mask[start_idx:start_idx + nodes_per_subgraph]

        sub_data = Data(x=sub_x, new_x=sub_new_x, edge_index=sub_edge_index, y=sub_y, adj=sub_adj,
                        train_mask=sub_train_mask, val_mask=sub_val_mask, test_mask=sub_test_mask)

        subgraphs.append(sub_data)

    color_print(f'!!!!! Finish node sampling')

    return subgraphs


def sampleCheck(node_groups, nodes_per_subgraph):
    cnt = 0
    wrong_num = 0
    for g in node_groups:
        if g.x.shape[0] != nodes_per_subgraph:
            color_print(g.x.shape[0], cnt)
            color_print(f'!!!!! Sampling check false')
            wrong_num = wrong_num + 1
        cnt = cnt + 1
    if not wrong_num:
        color_print(f'!!!!! Sampling check true')


def loadDataset(dataset, train_ratio):
    # 既然不用 DGL，我们将旧的数据集逻辑置空，重点保护 dgraph-fin
    graph_dgl = None

    if dataset == 'dgraph-fin':
        # 1. 读取假数据（或真数据）
        raw_data = np.load('./data/dgraph-fin.npz', allow_pickle=True)
        feat = torch.from_numpy(raw_data['x']).float()
        label = torch.from_numpy(raw_data['y']).long()
        edge_index = torch.from_numpy(raw_data['edge_index']).long().t()

        # 2. 划分数据集
        valid_indices = (label == 0) | (label == 1)
        nodes_idx = torch.where(valid_indices)[0].tolist()
        train_mask, val_mask, test_mask = splitDataset(nodes_idx, label, train_ratio)

        # 3. 封装为 PyG 对象
        graph_pyg = Data(x=feat, edge_index=edge_index, y=label,
                         train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        graph_pyg.new_x = feat

        color_print(f'DGraph-Fin 加载成功！特征维度: {feat.shape[1]}')

    else:
        # 其他数据集因为依赖 DGL，暂时跳过
        print(f'Dataset {dataset} requires DGL but DLL is missing. Skipping...')
        return None, None, None, None, None

    return graph_dgl, graph_pyg, train_mask, val_mask, test_mask


def mergeGraphDataList(args, graph_pyg, syn_relation_dict):
    # 此函数逻辑保持，已去掉 DGL 依赖
    x, y, new_x, edge_index = [], [], [], []
    adj_matrices = []
    ddpm_train_mask, ddpm_val_mask, ddpm_test_mask = [], [], []
    cum_num_nodes = 0

    new_x_ssupgcl = syn_relation_dict['graph_pyg_ssupgcl_new_x']

    for data in syn_relation_dict['syn_relation_list']:
        x.append(data.x)
        new_x.append(data.new_x)
        y.append(data.y)
        current_edge_index = data.edge_index + cum_num_nodes
        edge_index.append(current_edge_index)
        adj_matrices.append(data.adj)
        ddpm_train_mask.append(data.train_mask)
        ddpm_val_mask.append(data.val_mask)
        ddpm_test_mask.append(data.test_mask)
        cum_num_nodes += data.num_nodes

    node_per_subgraph = args.nodes_per_subgraph
    num_unselected = graph_pyg.x.shape[0] % node_per_subgraph

    new_x = torch.cat(new_x, dim=0)
    new_x = torch.cat([new_x, new_x_ssupgcl[-num_unselected:]], dim=0)
    x = torch.cat(x, dim=0)
    x = torch.cat([x, graph_pyg.x[-num_unselected:]], dim=0)
    y = torch.cat(y, dim=0)
    y = torch.cat([y, graph_pyg.y[-num_unselected:]], dim=0)
    edge_index = torch.cat(edge_index, dim=1)
    edge_index = torch.cat([edge_index, syn_relation_dict['unselected_edge_index']], dim=1)
    ddpm_train_mask = torch.cat(ddpm_train_mask, dim=0)
    ddpm_train_mask = torch.cat([ddpm_train_mask, graph_pyg.train_mask[-num_unselected:]], dim=0)
    ddpm_val_mask = torch.cat(ddpm_val_mask, dim=0)
    ddpm_val_mask = torch.cat([ddpm_val_mask, graph_pyg.val_mask[-num_unselected:]], dim=0)
    ddpm_test_mask = torch.cat(ddpm_test_mask, dim=0)
    ddpm_test_mask = torch.cat([ddpm_test_mask, graph_pyg.test_mask[-num_unselected:]], dim=0)

    merged_data = Data(x=x, edge_index=edge_index, y=y, train_mask=ddpm_train_mask, val_mask=ddpm_val_mask,
                       test_mask=ddpm_test_mask)
    return merged_data


def GDCAugment(graph_pyg_type, avg_degree):
    gdc = GDC(self_loop_weight=1, normalization_in='sym',
              normalization_out='col', diffusion_kwargs=dict(method='ppr', alpha=0.05),
              sparsification_kwargs=dict(method='threshold', avg_degree=avg_degree), exact=True)
    graph_pyg_type.transform = gdc
    gdc_transformed_data = gdc(graph_pyg_type)
    return gdc_transformed_data

# 这个函数由于深度依赖 DGL 的 heterograph，暂时注释掉
# def data4WFusionTrain(graph_pyg, graph_syn, graph_gdc_list):
#     ...
