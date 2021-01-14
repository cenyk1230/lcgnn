import torch

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
