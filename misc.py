from ogb.graphproppred import PygGraphPropPredDataset
from ogb.linkproppred import PygLinkPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


def create_dataset(name):
    if name.startswith("ogbn"):
        dataset = name[5:]
        if dataset in ["arxiv", "products", "papers100M", "proteins"]:
            return PygNodePropPredDataset(name)
        else:
            return Planetoid("./dataset", dataset, transform=T.NormalizeFeatures())
    elif name.startswith("ogbl"):
        return PygLinkPropPredDataset(name)
    elif name.startswith("ogbg"):
        return PygGraphPropPredDataset(name)
