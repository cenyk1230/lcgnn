import torch

PATH_DICT = {
    "reddit": "./dataset/Reddit/processed/data.pt",
    "flickr": "./dataset/Flickr/processed/data.pt",
    "yelp": "./dataset/Yelp/processed/data.pt",
}

num_classes_dict = dict(arxiv=40, products=47, papers100M=172)

class Data(object):
    def __init__(self, x, y):
        super(Data, self).__init__()
        self.x = x
        self.y = y
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
        self.num_classes = self.data.y.max().item() + 1

    def get_idx_split(self):
        return {
            "train": self.train_idx,
            "valid": self.val_idx,
            "test": self.test_idx,
        }

    def __getitem__(self, idx):
        return self.data
