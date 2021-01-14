import os
import argparse
import multiprocessing

import numpy as np
import scipy.sparse as sparse
import sklearn.preprocessing as preprocessing
from scipy.sparse import coo_matrix, csr_matrix, linalg
from tqdm import tqdm

from localgraphclustering import *
from misc import create_dataset
from torch_geometric.utils import (add_remaining_self_loops,
                                   to_scipy_sparse_matrix, to_undirected)


def calc(args):
    i, log_steps, num_iter, ego_size, weighted = args
    if i % log_steps == 0:
        print(i)
    output = approximate_PageRank_weighted(graphlocal, [i], iterations=num_iter) \
                if weighted else approximate_PageRank(graphlocal, [i], iterations=num_iter)
    if ego_size > 0:
        ego = np.array(output[0][:ego_size], dtype=np.int32)
        return ego, np.concatenate((ego, [-1] * (ego_size - len(ego))))
    else:
        ego = np.array(output[0], dtype=np.int32)
        return ego, None

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
    i, log_steps, num_iter, ego_size = args
    if i % log_steps == 0:
        print(i)
    node, ppr = approximate_PageRank(graphlocal, [i], iterations=num_iter, method="acl", normalize=False)
    d_inv = graphlocal.dn[node]
    d_inv[d_inv > 1.0] = 1.0
    ppr_d_inv = ppr * d_inv
    output = list(zip(node, ppr_d_inv, ppr))[:ego_size]
    node, ppr_d_inv, ppr = zip(*sorted(output, key=lambda x: x[1], reverse=True))
    assert node[0] == i
    node, ppr_d_inv, ppr = np.array(node, dtype=np.int32), np.array(ppr_d_inv, dtype=np.float), np.array(ppr, dtype=np.float)
    conds = my_sweep_cut(graphlocal, node)
    #cut, conductance = sweep_cut(graphlocal, (node, ppr), do_sort=False)
    #assert conds[len(cut)-1] == conductance 
    return node, ppr_d_inv, ppr, conds


def step1_multigraph(task, name, ego_size=128, num_iter=1000, log_steps=10000):
    dataset = create_dataset(name=f'{task}-{name}')
    data = dataset[0]

    N = data.num_nodes
    edge_index = data.edge_index.numpy()
    assert name == "proteins"

    num_graphs = data.edge_attr.size(-1)
    for j in range(3, num_graphs):
        adj = to_scipy_sparse_matrix(
            edge_index=data.edge_index,
            edge_attr=data.edge_attr[:, j],
            num_nodes=data.num_nodes).tocsr()

        global graphlocal
        graphlocal = GraphLocal.from_sparse_adjacency(adj)
        print('%d-th graphlocal generated' % j)

        ego_graphs = []
        ego_graphs_padding = []
        with multiprocessing.Pool(32) as pool:
            ego_graphs, ego_graphs_padding = zip(*pool.imap(calc, [(i, log_steps, num_iter, ego_size, True) for i in range(N)], chunksize=512))
        ego_graphs_padding = np.vstack(ego_graphs_padding).astype(np.int32)
        np.save(f'data/{name}-ego-graphs-{ego_size}-graph-{j}.npy', ego_graphs)
        np.save(f'data/{name}-ego-graphs-padding-{ego_size}-graph-{j}.npy', ego_graphs_padding)


def step1(task, name, ego_size=128, num_iter=1000, log_steps=10000):
    dataset = create_dataset(name=f'{task}-{name}')
    data = dataset[0]

    N = data.num_nodes
    edge_index = data.edge_index
    edge_index = to_undirected(edge_index)
    adj = csr_matrix((np.ones(edge_index.shape[1]), edge_index), shape=(N, N))

    global graphlocal
    graphlocal = GraphLocal.from_sparse_adjacency(adj)
    print('graphlocal generated')

    ego_graphs = []
    ego_graphs_padding = []
    with multiprocessing.Pool(16) as pool:
        ego_graphs, ego_graphs_padding = zip(*pool.imap(calc, [(i, log_steps, num_iter, ego_size, False) for i in range(N)], chunksize=512))
    np.save(f'data/{name}-ego-graphs-{ego_size}.npy', ego_graphs)
    if ego_size > 0:
        ego_graphs_padding = np.vstack(ego_graphs_padding).astype(np.int32)
        np.save(f'data/{name}-ego-graphs-padding-{ego_size}.npy', ego_graphs_padding)


def step1_local_clustering(task, name, ego_size=128, num_iter=1000, log_steps=10000):
    dataset = create_dataset(name=f'{task}-{name}')
    data = dataset[0]

    N = data.num_nodes
    edge_index = data.edge_index
    edge_index = to_undirected(edge_index)
    adj = csr_matrix((np.ones(edge_index.shape[1]), edge_index), shape=(N, N))

    global graphlocal
    graphlocal = GraphLocal.from_sparse_adjacency(adj)
    print('graphlocal generated')

    ego_graphs = []
    ego_graphs_padding = []
    with multiprocessing.Pool(32) as pool:
        ego_graphs, ppr_d_inv, ppr, conds = zip(*pool.imap(calc_local_clustering, [(i, log_steps, num_iter, ego_size) for i in range(N)], chunksize=512))
    np.save(f'data/{name}-lc-ego-graphs-{ego_size}.npy', ego_graphs)
    # np.save(f'data/{name}-lc-ppr_d_inv-{ego_size}.npy', ppr_d_inv)
    # np.save(f'data/{name}-lc-ppr-{ego_size}.npy', ppr)
    np.save(f'data/{name}-lc-conds-{ego_size}.npy', conds)
    #if ego_size > 0:
    #    ego_graphs_padding = np.vstack(ego_graphs_padding).astype(np.int32)
    #    np.save(f'data/{name}-ego-graphs-padding-{ego_size}.npy', ego_graphs_padding)

def calc2(args):
    adj, ego, ego_size = args
    ego_adj = adj[ego, :][:, ego].tocoo()
    return coo_matrix((ego_adj.data, (ego_adj.row, ego_adj.col)), shape=(ego_size, ego_size)).toarray().astype(np.bool)

def step2(task, name, ego_size=128):
    dataset = create_dataset(name=f'{task}-{name}')
    data = dataset[0]

    N = data.num_nodes
    edge_index = data.edge_index
    edge_index = to_undirected(edge_index)
    edge_index, _ = add_remaining_self_loops(edge_index)
    adj = csr_matrix((np.ones(edge_index.shape[1]), edge_index.numpy()), shape=(N, N))

    ego_graphs = np.load(f'data/{name}-ego-graphs-{ego_size}.npy', allow_pickle=True)

    with multiprocessing.Pool(16) as pool:
        ego_graphs_adj = list(pool.imap(calc2, ((adj, ego, ego_size) for ego in tqdm(ego_graphs)), chunksize=5120))
    ego_graphs_adj = np.stack(ego_graphs_adj)
    np.save(f'data/{name}-ego-graphs-adj-{ego_size}.npy', ego_graphs_adj)


def eigen_decomposision(n, k, laplacian, hidden_size, retry):
    if k <= 0:
        return np.zeros((n, hidden_size))
    laplacian = laplacian.astype("float32")
    ncv = min(n, max(2 * k + 1, 20))
    # follows https://stackoverflow.com/questions/52386942/scipy-sparse-linalg-eigsh-with-fixed-seed
    v0 = np.random.rand(n).astype("float32")
    for i in range(retry):
        try:
            s, u = linalg.eigsh(laplacian, k=k, which="LA", ncv=ncv, v0=v0)
        except sparse.linalg.eigen.arpack.ArpackError:
            ncv = min(ncv * 2, n)
            if i + 1 == retry:
                print("arpack error, retry=", i)
                sparse.save_npz("arpack_error_sparse_matrix.npz", laplacian)
                u = np.zeros((n, k))
                exit(0)
        else:
            break
    x = preprocessing.normalize(u, norm="l2")
    if hidden_size > k:
        x = np.concatenate((x, np.zeros((n, hidden_size - k))), axis=1)
    return x.astype("float32")


def _add_undirected_graph_positional_embedding(adj, n, hidden_size, retry=10):
    # We use eigenvectors of normalized graph laplacian as vertex features.
    # It could be viewed as a generalization of positional embedding in the
    # attention is all you need paper.
    # Recall that the eignvectors of normalized laplacian of a line graph are cos/sin functions.
    # See section 2.4 of http://www.cs.yale.edu/homes/spielman/561/2009/lect02-09.pdf
    norm = sparse.diags(
        adj.sum(axis=1).getA().squeeze(1).clip(1) ** -0.5, dtype=float
    )
    laplacian = norm * adj * norm
    k = min(n - 2, hidden_size)
    x = eigen_decomposision(n, k, laplacian, hidden_size, retry)
    return x

def calc3(args):
    adj, ego, ego_size, hidden_size = args
    gpe = _add_undirected_graph_positional_embedding(adj[ego, :][:, ego], len(ego), hidden_size)
    return np.concatenate((gpe, np.zeros((ego_size - len(ego), hidden_size))), axis=0)

def step3(task, name, ego_size=128, hidden_size=128):
    dataset = create_dataset(name=f'{task}-{name}')
    data = dataset[0]

    N = data.num_nodes
    edge_index = data.edge_index.numpy()
    edge_index = np.concatenate((edge_index, edge_index[::-1, :]), axis=1)
    adj = csr_matrix((np.ones(edge_index.shape[1]), edge_index), shape=(N, N))

    ego_graphs = np.load(f'data/{name}-ego-graphs-{ego_size}.npy', allow_pickle=True)

    print('load done!')

    with multiprocessing.Pool(16) as pool:
        ego_graphs_gpe = list(pool.imap(calc3, ((adj, ego, ego_size, hidden_size) for ego in tqdm(ego_graphs)), chunksize=5120))

    ego_graphs_gpe = np.stack(ego_graphs_gpe).astype(np.float32)
    np.save(f'data/{name}-ego-graphs-gpe-{ego_size}-{hidden_size}.npy', ego_graphs_gpe)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OGB (Preprocessing)')
    parser.add_argument('--task', type=str, default='ogbn')
    parser.add_argument('--dataset', type=str, default='arxiv')
    parser.add_argument('--ego_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_iter', type=int, default=1000)
    parser.add_argument('--log_steps', type=int, default=10000)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)

    if not os.path.exists('data'):
        os.makedirs('data')

    if args.step == 1:
        #step1(task=args.task, name=args.dataset, ego_size=args.ego_size, num_iter=args.num_iter, log_steps=args.log_steps)
        step1_local_clustering(task=args.task, name=args.dataset, ego_size=args.ego_size, num_iter=args.num_iter, log_steps=args.log_steps)
    elif args.step == 2:
        step2(task=args.task, name=args.dataset, ego_size=args.ego_size)
    elif args.step == 3:
        step3(task=args.task, name=args.dataset, ego_size=args.ego_size, hidden_size=args.hidden_size)
