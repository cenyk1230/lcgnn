from mydataset import SAINTDataset, MyMAGDataset, MyMAG240MDataset
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T

def create_dataset(name):
    if name.startswith("ogbn"):
        dataset = name[5:]
        if dataset in ["arxiv", "products", "papers100M", "proteins"]:
            return PygNodePropPredDataset(name)
        elif dataset == "mag":
            return MyMAGDataset("/home/yukuo/LCGNN-Homo-Hetero/data/ogbn_mag")
    elif name.startswith("saint"):
        dataset = name[6:]
        return SAINTDataset(dataset)
    elif name.startswith("lsc"):
        return MyMAG240MDataset("/home/yukuo/mag240m_kddcup2021")
