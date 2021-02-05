import argparse
import scipy.sparse as sp

import torch
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import to_undirected, to_scipy_sparse_matrix, dropout_adj

from misc import create_dataset

def save_embedding(model, dim, name, window):
    torch.save(model.embedding.weight.data.cpu(), f'data/{name}-embedding-{dim}-window-{window}.pt')


def main():
    parser = argparse.ArgumentParser(description='OGB (Node2Vec)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--task', type=str, default='ogbn')
    parser.add_argument('--dataset', type=str, default='arxiv')
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--walk_length', type=int, default=80)
    parser.add_argument('--context_size', type=int, default=20)
    parser.add_argument('--walks_per_node', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--dropedge_rate', type=float, default=0.4)
    parser.add_argument('--dump_adj_only', dest="dump_adj_only", action="store_true", help="dump adj matrix for proX")
    parser.set_defaults(dump_adj_only=False)
    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = create_dataset(name=f'{args.task}-{args.dataset}')
    data = dataset[0]
    if args.dataset == 'arxiv':
        data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    elif args.dataset == 'papers100M':
        data.edge_index, _ = dropout_adj(data.edge_index, p = args.dropedge_rate, num_nodes= data.num_nodes)
        data.edge_index = to_undirected(data.edge_index, data.num_nodes)

    if args.dump_adj_only:
        adj = to_scipy_sparse_matrix(data.edge_index)
        sp.save_npz(f'data/{args.name}-adj.npz', adj)
        return

    model = Node2Vec(data.edge_index, args.embedding_dim, args.walk_length,
                     args.context_size, args.walks_per_node,
                     sparse=True).to(device)

    loader = model.loader(batch_size=args.batch_size, shuffle=True,
                          num_workers=4)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(1, args.epochs + 1):
        for i, (pos_rw, neg_rw) in enumerate(loader):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()

            if (i + 1) % args.log_steps == 0:
                print(f'Epoch: {epoch:02d}, Step: {i+1:03d}/{len(loader)}, '
                      f'Loss: {loss:.4f}')

            if (i + 1) % 100 == 0:  # Save model every 100 steps.
                save_embedding(model, args.embedding_dim, args.dataset, args.context_size)
        save_embedding(model, args.embedding_dim, args.dataset, args.context_size)


if __name__ == "__main__":
    main()