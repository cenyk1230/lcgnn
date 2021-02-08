import argparse
import multiprocessing
import os

import numpy as np
from localgraphclustering import *
from scipy.sparse import csr_matrix
from torch_geometric.utils import to_undirected

from misc import create_dataset


def my_sweep_cut(g, node):
    vol_sum = 0.0
    in_edge = 0.0
    conds = np.zeros_like(node, dtype=np.float)
    for i in range(len(node)):
        idx = node[i]
        vol_sum += g.d[idx]
        denominator = min(vol_sum, g.vol_G - vol_sum)
        if denominator == 0.0:
            denominator = 1.0
        in_edge += 2*sum([g.adjacency_matrix[idx,prev] for prev in node[:i+1]])
        cut = vol_sum - in_edge
        conds[i] = cut/denominator
    return conds

def calc_local_clustering(args):
    i, log_steps, num_iter, ego_size, method = args
    if i % log_steps == 0:
        print(i)
    node, ppr = approximate_PageRank(graphlocal, [i], iterations=num_iter, method=method, normalize=False)
    d_inv = graphlocal.dn[node]
    d_inv[d_inv > 1.0] = 1.0
    ppr_d_inv = ppr * d_inv
    output = list(zip(node, ppr_d_inv))[:ego_size]
    node, ppr_d_inv = zip(*sorted(output, key=lambda x: x[1], reverse=True))
    assert node[0] == i
    node = np.array(node, dtype=np.int32)
    conds = my_sweep_cut(graphlocal, node)
    return node, conds

def step1_local_clustering(task, name, ego_size=128, num_iter=1000, log_steps=10000, num_workers=16, method='acl'):
    dataset = create_dataset(name=f'{task}-{name}')
    data = dataset[0]

    N = data.num_nodes
    edge_index = data.edge_index
    edge_index = to_undirected(edge_index)
    adj = csr_matrix((np.ones(edge_index.shape[1]), edge_index), shape=(N, N))

    global graphlocal
    graphlocal = GraphLocal.from_sparse_adjacency(adj)
    print('graphlocal generated')

    idx_split = dataset.get_idx_split()
    train_idx = idx_split["train"].cpu().numpy()
    valid_idx = idx_split["valid"].cpu().numpy()
    test_idx = idx_split["test"].cpu().numpy()

    with multiprocessing.Pool(num_workers) as pool:
        ego_graphs_train, conds_train = zip(*pool.imap(calc_local_clustering, [(i, log_steps, num_iter, ego_size, method) for i in train_idx], chunksize=512))

    with multiprocessing.Pool(num_workers) as pool:
        ego_graphs_valid, conds_valid = zip(*pool.imap(calc_local_clustering, [(i, log_steps, num_iter, ego_size, method) for i in valid_idx], chunksize=512))

    with multiprocessing.Pool(num_workers) as pool:
        ego_graphs_test, conds_test = zip(*pool.imap(calc_local_clustering, [(i, log_steps, num_iter, ego_size, method) for i in test_idx], chunksize=512))

    ego_graphs = []
    conds = []
    ego_graphs.extend(ego_graphs_train)
    ego_graphs.extend(ego_graphs_valid)
    ego_graphs.extend(ego_graphs_test)
    conds.extend(conds_train)
    conds.extend(conds_valid)
    conds.extend(conds_test)

    np.save(f"data/{name}-lc-ego-graphs-{ego_size}.npy", ego_graphs)
    np.save(f"data/{name}-lc-conds-{ego_size}.npy", conds)

def calc_inductive(args):
    i, log_steps, num_iter, ego_size, method = args
    if i % log_steps == 0:
        print(i)
    if graphlocal.dn[i] == 0:
        print('isolated node (testing):', i)
        node = np.array([i], dtype=np.int32)
        conds = np.array([1.0], dtype=np.float)
        return node, conds
    node, ppr_d_inv = approximate_PageRank(graphlocal, [i], iterations=num_iter, method=method, normalize=True)
    # d_inv = graphlocal.dn[node]
    # d_inv[d_inv > 1.0] = 1.0
    # ppr_d_inv = ppr * d_inv
    output = list(zip(node, ppr_d_inv))[:ego_size]
    node, ppr_d_inv = zip(*sorted(output, key=lambda x: x[1], reverse=True))
    node = list(node)
    if node[0] != i:
        if i not in node:
            node = [i] + node[:-1]
        else:
            idx = node.index(i)
            node = [i] + node[:idx] + node[idx+1:]
    assert node[0] == i
    node = np.array(node, dtype=np.int32)
    conds = my_sweep_cut(graphlocal, node)
    return node, conds

def calc_inductive_train(args):
    i, log_steps, num_iter, ego_size, method = args
    if i % log_steps == 0:
        print(i)
    if graphlocal_train.dn[i] == 0:
        print('isolated node (training):', i)
        node = np.array([i], dtype=np.int32)
        conds = np.array([1.0], dtype=np.float)
        return node, conds
    node, ppr_d_inv = approximate_PageRank(graphlocal_train, [i], iterations=num_iter, method=method, normalize=True)
    # d_inv = graphlocal_train.dn[node]
    # d_inv[d_inv > 1.0] = 1.0
    # ppr_d_inv = ppr * d_inv
    output = list(zip(node, ppr_d_inv))[:ego_size]
    node, ppr_d_inv = zip(*sorted(output, key=lambda x: x[1], reverse=True))
    node = list(node)
    if node[0] != i:
        if i not in node:
            node = [i] + node[:-1]
        else:
            idx = node.index(i)
            node = [i] + node[:idx] + node[idx+1:]
    assert node[0] == i
    node = np.array(node, dtype=np.int32)
    conds = my_sweep_cut(graphlocal_train, node)
    return node, conds

def step1_inductive(task, name, ego_size=128, num_iter=1000, log_steps=10000, num_workers=16, method='acl'):
    dataset = create_dataset(name=f'{task}-{name}')
    data = dataset[0]

    N = data.num_nodes
    edge_index = data.edge_index
    edge_index = to_undirected(edge_index)
    if hasattr(data, "edge_index_train"):
        edge_index_train = data.edge_index_train
        edge_index_train = to_undirected(edge_index_train)
    else:
        edge_index_train = edge_index
    adj = csr_matrix((np.ones(edge_index.shape[1]), edge_index), shape=(N, N))
    adj_train = csr_matrix((np.ones(edge_index_train.shape[1]), edge_index_train), shape=(N, N))

    idx_split = dataset.get_idx_split()
    train_idx = idx_split["train"].cpu().numpy()
    valid_idx = idx_split["valid"].cpu().numpy()
    test_idx = idx_split["test"].cpu().numpy()

    global graphlocal
    global graphlocal_train
    graphlocal = GraphLocal.from_sparse_adjacency(adj)
    graphlocal_train = GraphLocal.from_sparse_adjacency(adj_train)
    print('graphlocal generated')

    with multiprocessing.Pool(num_workers) as pool:
        ego_graphs_train, conds_train = zip(*pool.imap(calc_inductive_train, [(i, log_steps, num_iter, ego_size, method) for i in train_idx], chunksize=512))

    with multiprocessing.Pool(num_workers) as pool:
        ego_graphs_valid, conds_valid = zip(*pool.imap(calc_inductive, [(i, log_steps, num_iter, ego_size, method) for i in valid_idx], chunksize=512))

    with multiprocessing.Pool(num_workers) as pool:
        ego_graphs_test, conds_test = zip(*pool.imap(calc_inductive, [(i, log_steps, num_iter, ego_size, method) for i in test_idx], chunksize=512))

    ego_graphs = []
    conds = []
    ego_graphs.extend(ego_graphs_train)
    ego_graphs.extend(ego_graphs_valid)
    ego_graphs.extend(ego_graphs_test)
    conds.extend(conds_train)
    conds.extend(conds_valid)
    conds.extend(conds_test)

    if method == 'acl':
        np.save(f"data/{name}-lc-ego-graphs-{ego_size}.npy", ego_graphs)
        np.save(f"data/{name}-lc-conds-{ego_size}.npy", conds)
    else:
        np.save(f"data/{name}-lc-{method}-ego-graphs-{ego_size}.npy", ego_graphs)
        np.save(f"data/{name}-lc-{method}-conds-{ego_size}.npy", conds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LCGNN (Preprocessing)')
    parser.add_argument('--task', type=str, default='saint')
    parser.add_argument('--dataset', type=str, default='flickr')
    parser.add_argument('--ego_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_iter', type=int, default=1000)
    parser.add_argument('--log_steps', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--method', type=str, default='acl')
    parser.add_argument('--num_workers', type=int, default=16)
    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)

    if not os.path.exists('data'):
        os.makedirs('data')

    if args.task == 'ogbn':
        step1_local_clustering(task=args.task, name=args.dataset, ego_size=args.ego_size, num_iter=args.num_iter, log_steps=args.log_steps, num_workers=args.num_workers, method=args.method)
    else:
        step1_inductive(task=args.task, name=args.dataset, ego_size=args.ego_size, num_iter=args.num_iter, log_steps=args.log_steps, num_workers=args.num_workers, method=args.method)
