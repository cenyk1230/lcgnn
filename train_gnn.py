import argparse
import os

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from ogb.nodeproppred import DglNodePropPredDataset
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from gnn import GNNModel
# from line_profiler import LineProfiler
from mydataset import MyNodePropPredDataset, SAINTDataset
from optim_schedule import NoamOptim


class NodeClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, dgl_data, ego_graphs, cut):
        self.graph = dgl_data[0]
        self.label = dgl_data[1]
        
        self.ego_graphs = ego_graphs
        self.cut = cut

    def __getitem__(self, idx):
        nids = self.ego_graphs[idx]
        subg = self.graph.subgraph(nids)
        nfeat = [subg.ndata['feat'], self.cut[idx]]
        subg.ndata['feat'] = torch.cat(nfeat, dim=-1)
        label = self.label[nids[0]]

        return subg, label

    def __len__(self):
        return len(self.ego_graphs)

def batcher():
    def batcher_gnn(batch):
        graph, label = zip(*batch)
        graph = dgl.batch(graph)
        label = torch.stack(label).long()

        return graph, label

    return batcher_gnn


def accuracy(y_true, y_pred):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum()
    return correct / len(y_true)

def multilabel_f1(y_true, y_pred, sigmoid=False):
    if sigmoid:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    else:
        y_pred[y_pred > 0] = 1
        y_pred[y_pred <= 0] = 0
    preds = y_pred.cpu().detach()
    labels = y_true.cpu().float()
    return f1_score(labels, preds, average="micro")

def train(model, loader, device, optimizer, args):
    model.train()

    total_loss = 0
    for batch in tqdm(loader, desc="Iteration"):
        optimizer.zero_grad()

        batch = [x.to(device) for x in batch]
        g, y = batch

        out = model(g)

        if args.dataset in ['ppi', 'yelp', 'amazon']:
            loss = F.binary_cross_entropy_with_logits(out, y.float())
        else:
            loss = F.cross_entropy(out, y.squeeze(1))

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)

@torch.no_grad()
def test(model, loader, device, args):
    model.eval()

    y_pred, y_true = [], []

    for batch in tqdm(loader, desc="Iteration"):
        batch = [x.to(device) for x in batch]
        g, y = batch

        out = model(g)

        y_pred.append(out)
        y_true.append(y)

    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)

    if args.dataset in ['ppi', 'yelp', 'amazon']:
        loss = F.binary_cross_entropy_with_logits(y_pred, y_true.float()).item()
        metric = multilabel_f1(y_true, y_pred, sigmoid=False)
    else:
        loss = F.cross_entropy(y_pred, y_true.squeeze(1)).item()
        metric = accuracy(y_true, y_pred)

    return metric, loss


def get_exp_name(dataset, para_dic, input_exp_name):
    para_name = '_'.join([dataset] + [key + str(value) for key, value in para_dic.items()])
    exp_name = para_name + '_' + input_exp_name

    return exp_name

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def main():
    parser = argparse.ArgumentParser(description='OGBN (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='arxiv')
    parser.add_argument('--model', type=str, default='gcn')
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--ego_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--input_dropout', type=float, default=0.2)
    parser.add_argument('--hidden_dropout', type=float, default=0.4)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--early_stopping', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--eval_batch_size', type=int, default=512)
    parser.add_argument('--batch_norm', type=int, default=1)
    parser.add_argument('--residual', type=int, default=1)
    parser.add_argument('--linear_layer', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument("--optimizer", type=str, default='adamw', choices=['adam', 'adamw'], help="optimizer")
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--exp_name', type=str, default='')
    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    para_dic = {'': args.model, 'nl': args.num_layers, 'nh': args.num_heads, 'es': args.ego_size, 'hs': args.hidden_size,
                'id': args.input_dropout, 'hd': args.hidden_dropout, 'bs': args.batch_size, 'op': args.optimizer, 
                'lr': args.lr, 'wd': args.weight_decay, 'bn': args.batch_norm, 
                'rs': args.residual, 'll': args.linear_layer, 'sd': args.seed}
    para_dic['warm'] = args.warmup
    exp_name = get_exp_name(args.dataset, para_dic, args.exp_name)

    wandb_name = exp_name.replace('_sd'+str(args.seed), '')
    wandb.init(name=wandb_name, project="lcgnn")
    wandb.config.update(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if args.dataset == 'papers100M':
        dataset = MyNodePropPredDataset(name=args.dataset)
    elif args.dataset in ['ppi', 'flickr', 'reddit', 'yelp', 'amazon']:
        dataset = SAINTDataset(name=args.dataset)
    else:
        dataset = DglNodePropPredDataset(name=f'ogbn-{args.dataset}')

    split_idx = dataset.get_idx_split()
    train_idx = set(split_idx['train'].cpu().numpy())
    valid_idx = set(split_idx['valid'].cpu().numpy())
    test_idx = set(split_idx['test'].cpu().numpy())

    ego_graphs_unpadded = np.load(f'data/{args.dataset}-lc-ego-graphs-{args.ego_size}.npy', allow_pickle=True)
    conds_unpadded = np.load(f'data/{args.dataset}-lc-conds-{args.ego_size}.npy', allow_pickle=True)

    ego_graphs_train, ego_graphs_valid, ego_graphs_test = [], [], []
    cut_train, cut_valid, cut_test = [], [], []

    for i, ego_graph in enumerate(ego_graphs_unpadded):
        idx = ego_graph[0]
        assert len(ego_graph) == len(conds_unpadded[i])
        cut_position = np.argmin(conds_unpadded[i])
        cut = torch.zeros(len(ego_graph), dtype=torch.float32)
        cut[:cut_position+1] = 1.0
        cut = cut.unsqueeze(1)
        if idx in train_idx:
            ego_graphs_train.append(ego_graph)
            cut_train.append(cut)
        elif idx in valid_idx:
            ego_graphs_valid.append(ego_graph)
            cut_valid.append(cut)
        elif idx in test_idx:
            ego_graphs_test.append(ego_graph)
            cut_test.append(cut)
        else:
            print(f"{idx} not in train/valid/test idx")

    num_classes = dataset.num_classes

    if isinstance(dataset, DglNodePropPredDataset):
        data = dataset[0]
        graph = dgl.remove_self_loop(data[0])
        graph = dgl.add_self_loop(graph)
        if args.dataset == 'arxiv' or args.dataset == 'papers100M':
            temp_graph = dgl.to_bidirected(graph)
            temp_graph.ndata['feat'] = graph.ndata['feat']
            graph = temp_graph
        data = (graph, data[1].long())

        graph = data[0]
        graph.ndata['labels'] = data[1]
    elif isinstance(dataset, SAINTDataset):
        data = dataset[0]
        edge_index = data.edge_index
        graph = dgl.DGLGraph((edge_index[0], edge_index[1]))
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
        graph.ndata['feat'] = data.x
        label = data.y
        if len(label.shape) == 1:
            label = label.unsqueeze(1)
        data = (graph, label)
    else:
        raise NotImplementedError

    train_dataset = NodeClassificationDataset(data, ego_graphs_train, cut_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=batcher(), pin_memory=True)

    valid_dataset = NodeClassificationDataset(data, ego_graphs_valid, cut_valid)
    valid_loader = DataLoader(valid_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=batcher(), pin_memory=True)

    test_dataset = NodeClassificationDataset(data, ego_graphs_test, cut_test)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=batcher(), pin_memory=True)

    model = GNNModel(conv_type=args.model, input_size=graph.ndata['feat'].shape[1]+1, hidden_size=args.hidden_size, num_layers=args.num_layers, 
                     num_classes=num_classes, batch_norm=args.batch_norm, residual=args.residual, idropout=args.input_dropout, 
                     dropout=args.hidden_dropout, linear_layer=args.linear_layer, num_heads=args.num_heads).to(device)

    wandb.watch(model, log='all')

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model parameters:', pytorch_total_params)

    if not os.path.exists('saved'):
        os.mkdir('saved')

    model.reset_parameters()

    if args.load_path:
        model.load_state_dict(torch.load(args.load_path, map_location='cuda:0'))

        valid_acc, valid_loss = test(model, valid_loader, device, args)
        valid_output = f'Valid: {100 * valid_acc:.2f}% '

        cor_train_acc, _ = test(model, train_loader, device, args)

        cor_test_acc, cor_test_loss = test(model, test_loader, device, args)
        train_output = f'Train: {100 * cor_train_acc:.2f}%, '
        test_output = f'Test: {100 * cor_test_acc:.2f}%'

        print(train_output + valid_output + test_output)
        return

    best_val_acc = 0
    cor_train_acc = 0
    cor_test_acc = 0
    patience = 0

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError
    if args.warmup > 0:
        optimizer = NoamOptim(optimizer, args.hidden_size if args.hidden_size > 0 else data.x.size(1), n_warmup_steps=args.warmup, init_lr=args.lr)

    for epoch in range(1, 1 + args.epochs):
        # lp = LineProfiler()
        # lp_wrapper = lp(train)
        # loss = lp_wrapper(model, train_loader, device, optimizer, args)
        # lp.print_stats()
        loss = train(model, train_loader, device, optimizer, args)

        train_output = valid_output = test_output = ''
        if epoch % args.log_steps == 0:
            valid_acc, valid_loss = test(model, valid_loader, device, args)
            valid_output = f'Valid: {100 * valid_acc:.2f}% '

            if valid_acc > best_val_acc:
                best_val_acc = valid_acc
                # cor_train_acc, _ = test(model, train_loader, device, args)
                cor_test_acc, cor_test_loss = test(model, test_loader, device, args)
                # train_output = f'Train: {100 * cor_train_acc:.2f}%, '
                test_output = f'Test: {100 * cor_test_acc:.2f}%'
                patience = 0
                try:
                    torch.save(model.state_dict(), 'saved/' + exp_name + '.pt')
                    wandb.save('saved/' + exp_name + '.pt')
                except FileNotFoundError as e:
                    print(e)
            else:
                patience += 1
                if patience >= args.early_stopping:
                    print('Early stopping...')
                    break
            # 'cor_train_acc': cor_train_acc, 
            wandb.log({'Train Loss': loss, 'Valid Acc': valid_acc, 'best_val_acc': best_val_acc, 
                        'cor_test_acc': cor_test_acc, 'LR': get_lr(optimizer),
                        'Valid Loss': valid_loss, 'cor_test_loss': cor_test_loss})
        else:
            wandb.log({'Train Loss': loss, 'LR': get_lr(optimizer)})
        # train_output + 
        print(f'Epoch: {epoch:02d}, '
              f'Loss: {loss:.4f}, ' + 
              valid_output + test_output)


if __name__ == "__main__":
    main()
