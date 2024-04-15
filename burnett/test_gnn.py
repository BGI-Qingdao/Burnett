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
import torch.nn.functional as F
import networkx as nx
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io as strucio
import biotite.database.rcsb as rcsb
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


def ac_dist(file_path):
    '''
    空间中2个氨基酸集团的Ca原子（一般用Ca原子来计算接触）的空间距离小于8Å（Å是距离单位）的时候，我们认定这两个氨基酸是处于接触contact状态。
    简单讲，就是通过距离来判断是否接触，推断是否能够发生反应。
    :return:
    '''
    stack = strucio.load_structure(file_path)
    # Filter only CA atoms
    stack = stack[:, stack.atom_name == "CA"]
    # Calculate distance between first and second CA in first frame
    array = stack[0]
    print("Atom to atom:", struc.distance(array[0], array[1]))
    # Calculate distance between the first atom
    # and all other CA atoms in the array
    print("Array to atom:")
    array = stack[0]
    print(struc.distance(array[0], array))
    # Calculate pairwise distances between the CA atoms in the first frame
    # and the CA atoms in the second frame
    print("Array to array:")
    print(struc.distance(stack[0], stack[1]))
    # Calculate the distances between all CA atoms in the stack
    # and the first CA atom in the first frame
    # The resulting array is too large, therefore only the shape is printed
    print("Stack to atom:")
    print(struc.distance(stack, stack[0, 0]).shape)
    # And finally distances between two adjacent CA in the first frame
    array = stack[0]
    print("Adjacent CA distances")
    print(struc.distance(array[:-1], array[1:]))


def cal_angles(array):
    # Calculate angle between first 3 CA atoms in first frame
    # (in radians)
    print("Angle:", struc.angle(array[0], array[1], array[2]))
    # Calculate dihedral angle between first 4 CA atoms in first frame
    # (in radians)
    print("Dihedral angle:", struc.dihedral(array[0], array[1], array[2], array[4]))


def contact_matrix():
    ac_contact_matrix = np.array()
    return ac_contact_matrix


def construct_graph(structure):
    array = structure.get_array(0)  # <class 'biotite.structure.AtomArray'>
    # 获取原子坐标
    coords = array.coord
    # 添加节点和节点特征
    node_features = torch.tensor(coords, dtype=torch.float)
    # 构建图
    graph = nx.Graph()
    # TODO: 不要使用原子，使用氨基酸残基
    # 1. 定义氨基酸坐标coordinates: CA的坐标
    # 2. 定义氨基酸间的距离：CA的空间距离
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
    return graph, data


# plotting graph
def plot(graph):
    # pos = nx.get_node_attributes(G, 'pos')
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=300, node_color='skyblue', font_size=8)
    plt.savefig('network.png')


def plot2(H):
    node_colors = nx.get_node_attributes(H, "color").values()
    colors = list(node_colors)
    node_sizes = nx.get_node_attributes(H, "size").values()
    sizes = list(node_sizes)
    #Plotting Graph
    nx.draw(H, with_labels=True, node_color=colors, node_size=sizes)


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


def visualize_(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()


def visualize(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    else:
        nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                         node_color=color, cmap="Set2")


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


import mindspore as ms
from mindspore import Tensor, ops
from mindspore import numpy as msnp

def rigidFrom3Points(crd):
    """ Transform the coordinates formulation. """
    v1 = crd[:, 2] - crd[:, 1]
    v2 = crd[:, 0] - crd[:, 1]
    e1 = v1 / v1.norm(axis=-1, keep_dims=True)
    u2 = v2 - e1 * ops.Einsum('ij,ij->i')((e1, v2))[:, None]
    e2 = u2 / u2.norm(axis=-1, keep_dims=True)
    e3 = msnp.cross(e1, e2, axisc=-1)
    R = ops.Concat(axis=-2)((e1[:, None], e2[:, None], e3[:, None]))
    t = crd[:, 1][:, None]
    new_crd = ops.Concat(axis=-2)((R, t))
    return new_crd


if __name__ == '__main__':
    fn = '/home/share/huadjyin/home/fanguangyi/liyao1/evo_inter/data/pro1.pdb'
    pdb_file = pdb.PDBFile.read(fn)
    structure = pdb_file.get_structure()

    # 构建图
    graph, data = construct_graph(structure)

    plot(graph)

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


    np.random.seed(1)
    BATCHES = 2
    DIMENSIONS = 3
    ATOMS = 3
    # 定义原始坐标
    origin_crd = Tensor(np.random.random((BATCHES, ATOMS, DIMENSIONS)), ms.float32)
    print('The original coordinates is: \n{}'.format(origin_crd))
    trans_crd = rigidFrom3Points(origin_crd)
    print('The transformed coordinates is: \n{}'.format(trans_crd))
