import torch_geometric.datasets
from torch_geometric.data import Data
import torch_geometric.transforms as T
import torch
import sys
import networkx as nx
import os
import numpy as np
import scipy.sparse as sp
import torch_geometric.datasets
from incidence_matrix import get_faces, incidence_matrices

from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.utils import (
    negative_sampling,
)

class OGBL:
    def __init__(self, name):
        self.name = name
        self.dataset = PygLinkPropPredDataset(name=name, root='./data')
        self.num_classes = -99999 # stub, unused
        self.y = torch.arange(self.dataset[0].num_nodes) # used to count nodes in compute_hodge_matrix
        self.x = torch.nn.functional.one_hot(self.y).float() # homogeneous nodes
        #self.x = torch.zeros(self.dataset[0].num_nodes)[:, None].float() # homogeneous nodes
        #self.x = torch.randn(self.dataset[0].num_nodes, 32) # heterogeneous nodes
        #self.x = torch.arange(self.dataset[0].num_nodes) # indices to embedding table
        self.precompute_()

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.total_edges = self.total_edges.to(device)
        self.total_edges_y = self.total_edges_y.to(device)
        return self

    def precompute_(self):
        splits = self.dataset.get_edge_split()
        self.train_edges = splits['train']['edge']
        self.train_edges_false = negative_sampling(edge_index=self.train_edges.T,
                                              num_nodes=self.dataset._data.num_nodes,
                                              num_neg_samples=self.train_edges.shape[0]).T
        self.val_edges, self.val_edges_false = splits['valid']['edge'], splits['valid']['edge_neg']
        self.test_edges, self.test_edges_false = splits['test']['edge'], splits['test']['edge_neg']

        self.edge_index = torch.cat([self.train_edges, self.train_edges_false], dim=0).T
        self.total_edges = torch.cat((self.train_edges,self.train_edges_false,self.val_edges,self.val_edges_false,self.test_edges,self.test_edges_false))
        self.train_pos,self.train_neg = len(self.train_edges),len(self.train_edges_false)
        self.val_pos, self.val_neg = len(self.val_edges), len(self.val_edges_false)
        self.test_pos, self.test_neg = len(self.test_edges), len(self.test_edges_false)
        self.total_edges_y = torch.cat((torch.ones(len(self.train_edges)), torch.zeros(len(self.train_edges_false)), torch.ones(len(self.val_edges)), torch.zeros(len(self.val_edges_false)),torch.ones(len(self.test_edges)), torch.zeros(len(self.test_edges_false)))).long()

    def get_edges_split(self):
        return self.train_edges, self.train_edges_false, self.val_edges, self.val_edges_false, self.test_edges, self.test_edges_false


def loaddatas(d_name):
    if d_name in ["PPI"]:
        dataset = torch_geometric.datasets.PPI('./data/' + d_name)
    elif d_name == 'Cora':
        dataset = torch_geometric.datasets.Planetoid('./data/'+d_name,d_name,transform=T.NormalizeFeatures())
    elif d_name in ['Citeseer', 'PubMed']:
        dataset = torch_geometric.datasets.Planetoid('./data/' + d_name, d_name)
    elif d_name in ["Computers", "Photo"]:
        dataset = torch_geometric.datasets.Amazon('./data/'+d_name,d_name)
    elif d_name in ['ogbl-ddi']:
        dataset = OGBL(d_name)
    else:
        raise ValueError('Unknown dataset: {}'.format(d_name))

    return dataset


def get_edges_split(data, val_prop = 0.2, test_prop = 0.2):
    if isinstance(data, OGBL):
        return data.get_edges_split()

    g = nx.Graph()
    g.add_nodes_from([i for i in range(len(data.y))])
    _edge_index_ = np.array((data.edge_index))
    edge_index_ = [(_edge_index_[0, i], _edge_index_[1, i]) for i in
                        range(np.shape(_edge_index_)[1])]
    g.add_edges_from(edge_index_)
    adj = nx.adjacency_matrix(g)

    return get_adj_split(adj,val_prop = val_prop, test_prop = test_prop)


def get_adj_split(adj, val_prop=0.05, test_prop=0.1):
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    # get tn edges
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false


def compute_hodge_matrix(data, sample_data_edge_index):
    g = nx.Graph()
    g.add_nodes_from([i for i in range(len(data.y))])
    edge_index_ = np.array((sample_data_edge_index))
    edge_index = [(edge_index_[0, i], edge_index_[1, i]) for i in
                        range(np.shape(edge_index_)[1])]
    g.add_edges_from(edge_index)

    edge_to_idx = {edge: i for i, edge in enumerate(g.edges)}

    B1, B2 = incidence_matrices(g, sorted(g.nodes), sorted(g.edges), get_faces(g), edge_to_idx)

    return B1, B2


if __name__ == '__main__':
    ddi = loaddatas('ogbl-ddi')
    cora = loaddatas('Cora')