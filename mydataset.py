import os.path as osp
import time
import numpy as np
import torch

PATH_DICT = {
    "reddit": "./dataset/Reddit/processed/data.pt",
    "flickr": "./dataset/Flickr/processed/data.pt",
    "yelp": "./dataset/Yelp/processed/data.pt",
    "amazon": "./dataset/AmazonSaint/processed/data.pt",
}

num_classes_dict = dict(products=47, papers100M=172, mag240M=153)

class Data(object):
    def __init__(self, x, y, edge_index=None):
        super(Data, self).__init__()
        self.x = x
        self.y = y
        self.edge_index = edge_index
        self.num_nodes = x.shape[0]

class MyNodePropPredDataset(object):
    
    def __init__(self, name):
        super(MyNodePropPredDataset, self).__init__()
        self.name = name
        self.load_data()
    
    def __getitem__(self, idx):
        assert idx == 0
        return self.data

    def load_data(self):
        x = torch.load(f'data/ogbn-{self.name}-x.pt')
        y = torch.load(f'data/ogbn-{self.name}-y.pt')
        self.data = Data(x, y)
        self.num_classes = num_classes_dict[self.name]
        self.idx_split = torch.load(f'data/ogbn-{self.name}-split.pt')
    
    def get_idx_split(self):
        return self.idx_split


class SAINTDataset(object):
    def __init__(self, name):
        self.name = name
        self._load(PATH_DICT[self.name])

    def _load(self, path):
        self.data = torch.load(path)
        self.data.num_nodes = self.data.x.shape[0]
        self.train_idx = self.data.train_mask.nonzero().squeeze(1)
        self.val_idx = self.data.val_mask.nonzero().squeeze(1)
        self.test_idx = self.data.test_mask.nonzero().squeeze(1)
        if len(self.data.y.shape) == 1:
            self.num_classes = self.data.y.max().item() + 1
        else:
            self.num_classes = self.data.y.shape[1]

    def get_idx_split(self):
        return {
            "train": self.train_idx,
            "valid": self.val_idx,
            "test": self.test_idx,
        }

    def __getitem__(self, idx):
        return self.data

class MyMAG240MDataset(object):
    def __init__(self, data_dir, in_memory=False):
        self.data_dir = data_dir
        self.in_memory = in_memory
        self.load_data()

    def __getitem__(self, idx):
        assert idx == 0
        return self.data

    def get_idx_split(self):
        return self.idx_split

    def load_data(self):
        x = torch.from_numpy(np.load(osp.join(self.data_dir, 'processed/paper/node_feat.npy'), mmap_mode=None if self.in_memory else 'r'))
        y = torch.from_numpy(np.load(osp.join(self.data_dir, 'processed/paper/node_label.npy')))
        edge_index = torch.from_numpy(np.load(osp.join(self.data_dir, 'processed/paper___cites___paper/edge_index.npy')))
        self.data = Data(x, y, edge_index)
        self.num_classes = num_classes_dict["mag240M"]
        self.idx_split = torch.load(osp.join(self.data_dir, 'split_dict.pt'))
