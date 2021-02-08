from mydataset import SAINTDataset
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T


def create_dataset(name):
    if name.startswith("ogbn"):
        dataset = name[5:]
        if dataset in ["arxiv", "products", "papers100M", "proteins"]:
            return PygNodePropPredDataset(name)
    elif name.startswith("saint"):
        dataset = name[6:]
        return SAINTDataset(dataset)
