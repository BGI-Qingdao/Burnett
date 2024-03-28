#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date: Created on 27 Mar 2024 16:03
# @Author: Yao LI
# @File: evo_fish/test_gnn.py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, Data
import networkx as nx
import biotite.structure.io.pdb as pdb
%matplotlib inline
from sklearn.manifold import TSNE


'''
# Get the first atom
atom = array[0]
# Get a subarray containing the first and third atom
subarray = array[[0,2]]
# Get a subarray containing a range of atoms using slices
subarray = array[100:200]
# Filter all carbon atoms in residue 1
subarray = array[(array.element == "C") & (array.res_id == 1)]
# Filter all atoms where the X-coordinate is smaller than 2
subarray = array[array.coord[:,0] < 2]
mask = (array.atom_name != "C")
sub_array = array[mask]
'''


def construct_graph(struc):
    array = struc.get_array(0)  # <class 'biotite.structure.AtomArray'>
    # 获取原子坐标
    coords = array.coord
    # 添加节点和节点特征
    node_features = torch.tensor(coords, dtype=torch.float)
    # 构建图
    graph = nx.Graph()
    # TODO: 不要使用原子，使用氨基酸残基
    # 1. 定义氨基酸坐标coordinates
    # 2. 定义氨基酸间的距离

    # 1添加原子节点
    num_atoms = int(coords.shape[0])
    for i in range(num_atoms):
        graph.add_node(i, pos=coords[i])
    # 2计算原子之间的距离，并添加边
    threshold = 40
    for i in range(num_atoms):
        for j in range(i+1, num_atoms):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < threshold:  # 设置一个阈值，如果两个原子之间的距离小于阈值，则认为它们之间有连接关系
                graph.add_edge(i, j)
    # 2.2获取边索引
    edge_index = torch.tensor(list(graph.edges()), dtype=torch.long).t().contiguous()
    # 创建一个 Data 对象，用于存储图数据
    # data = Data(x=node_features, edge_index=edge_index)
    data = Data(x=node_features, edge_index=edge_index, train_mask=2708, val_mask=[2708], test_mask=2708)
    return data


# plotting graph
def plot(graph):
    # pos = nx.get_node_attributes(G, 'pos')
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=300, node_color='skyblue', font_size=8)
    plt.show()


# 定义一个简单的图神经网络模型
class ProteinGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ProteinGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return torch.log_softmax(x, dim=1)


def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()


def train():
    model.train()
    optimizer.zero_grad()
    # out = model(data.x, data.edge_index)
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


def test():
    model.eval()
    # out = model(data.x, data.edge_index)
    out = model(data)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc


if __name__ == '__main__':
    fn = 'pro1.pdb'
    pdb_file = pdb.PDBFile.read(fn)
    struc = pdb_file.get_structure()

    # 构建图
    data = construct_graph(struc)

    # 初始化图神经网络模型
    num_classes = 3  # TODO: 不知道需要输出多少类！
    model = ProteinGNN(input_dim=3, hidden_dim=64, output_dim=num_classes)
    print(model.eval())

    # 将模型应用于图数据
    output = model(data)

    # plot untrained GCN network
    visualize(output, color=data.y)

    # Train model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, 101):
        loss = train()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
