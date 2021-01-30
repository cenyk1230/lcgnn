import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

import wandb
from mydataset import MyNodePropPredDataset, SAINTDataset
# from line_profiler import LineProfiler
from ogb.nodeproppred import PygNodePropPredDataset
from optim_schedule import NoamOptim, LinearOptim
from transformer import TransformerModel


class NodeClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, ego_graphs, pe, args, num_classes, adj=None, cut=None):
        super(NodeClassificationDataset).__init__()
        self.x = x
        self.y = y
        self.ego_graphs = ego_graphs
        self.pe = pe
        self.args = args
        self.num_classes = num_classes
        self.adj = adj
        self.cut = cut

    def __len__(self):
        return len(self.ego_graphs)

    def __getitem__(self, idx):
        return idx

def batcher(dataset):
    def batcher_dev(idx):
        idx = torch.LongTensor(idx)
        src = dataset.ego_graphs[idx]
        batch_idx = src[:, 0]
        src_padding = (src == -1)
        src[src_padding] = 0
        shape = src.shape

        src_mask = [torch.repeat_interleave(dataset.adj[idx], dataset.args.num_heads, dim=0)] if dataset.adj is not None else []
        pe_batch = [dataset.pe[src.view(-1)].view(shape[0], shape[1], -1)] if dataset.pe is not None else []

        src = dataset.x[src.view(-1)].view(shape[0], shape[1], -1)
        y = dataset.y.squeeze(1)[batch_idx].long()

        cut = dataset.cut[idx].unsqueeze(-1)

        src = torch.cat((src, cut), dim=-1)

        return [src, src_padding, batch_idx, y] + src_mask + pe_batch

    return batcher_dev

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
        src, src_padding, _, y = batch[:4]
        src_mask = batch[4] if args.mask else None
        pe_batch = batch[-1] if args.pe_type else None

        out = model(src, src_mask=src_mask, padding=src_padding, pe=pe_batch)

        if args.dataset in ['ppi', 'yelp', 'amazon']:
            loss = F.binary_cross_entropy_with_logits(out, y.float())
        else:
            loss = F.cross_entropy(out, y)
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
        src, src_padding, batch_idx, y = batch[:4]
        src_mask = batch[4] if args.mask else None
        pe_batch = batch[-1] if args.pe_type else None

        out = model(src, src_mask=src_mask, padding=src_padding, pe=pe_batch)

        y_pred.append(out)
        y_true.append(y)

    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)

    if args.dataset in ['ppi', 'yelp', 'amazon']:
        loss = F.binary_cross_entropy_with_logits(y_pred, y_true.float()).item()
        metric = multilabel_f1(y_true, y_pred, sigmoid=False)
    else:
        loss = F.cross_entropy(y_pred, y_true).item()
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
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--ego_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--input_dropout', type=float, default=0.2)
    parser.add_argument('--hidden_dropout', type=float, default=0.4)
    parser.add_argument('--weight_decay', type=float, default=0.005)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--early_stopping', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--eval_batch_size', type=int, default=2048)
    parser.add_argument('--layer_norm', type=int, default=0)
    parser.add_argument('--src_scale', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--pe_type', type=int, default=0)
    parser.add_argument('--mask', type=int, default=0)
    parser.add_argument("--optimizer", type=str, default='adamw', choices=['adam', 'adamw'], help="optimizer")
    parser.add_argument("--scheduler", type=str, default='noam', choices=['noam', 'linear'], help="scheduler")
    parser.add_argument("--method", type=str, default='acl', choices=['acl', 'l1reg'], help="method for local clustering")
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--exp_name', type=str, default='')
    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    para_dic = {'nl': args.num_layers, 'nh': args.num_heads, 'es': args.ego_size, 'hs': args.hidden_size,
                'id': args.input_dropout, 'hd': args.hidden_dropout, 'bs': args.batch_size, 'pe': args.pe_type, 
                'op': args.optimizer, 'lr': args.lr, 'wd': args.weight_decay,
                'ln': args.layer_norm, 'sc': args.src_scale, 'sd': args.seed, 'md': args.method}
    para_dic['warm'] = args.warmup
    para_dic['mask'] = args.mask
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
        dataset = PygNodePropPredDataset(name=f'ogbn-{args.dataset}')

    split_idx = dataset.get_idx_split()
    train_idx = set(split_idx['train'].cpu().numpy())
    valid_idx = set(split_idx['valid'].cpu().numpy())
    test_idx = set(split_idx['test'].cpu().numpy())

    if args.method != "acl":
        ego_graphs_unpadded = np.load(f'data/{args.dataset}-lc-{args.method}-ego-graphs-{args.ego_size}.npy', allow_pickle=True)
        conds_unpadded = np.load(f'data/{args.dataset}-lc-{args.method}-conds-{args.ego_size}.npy', allow_pickle=True)
    else:
        tmp_ego_size = 256 if args.dataset == 'products' else args.ego_size
        ego_graphs_unpadded = np.load(f'data/{args.dataset}-lc-ego-graphs-{tmp_ego_size}.npy', allow_pickle=True)
        conds_unpadded = np.load(f'data/{args.dataset}-lc-conds-{tmp_ego_size}.npy', allow_pickle=True)

    ego_graphs_train, ego_graphs_valid, ego_graphs_test = [], [], []
    cut_train, cut_valid, cut_test = [], [], []

    for i, x in enumerate(ego_graphs_unpadded):
        idx = x[0]
        assert len(x) == len(conds_unpadded[i])
        if len(x) > args.ego_size:
            x = x[:args.ego_size]
            conds_unpadded[i] = conds_unpadded[i][:args.ego_size]
        ego_graph = -np.ones(args.ego_size, dtype=np.int32)
        ego_graph[:len(x)] = x
        cut_position = np.argmin(conds_unpadded[i])
        cut = np.zeros(args.ego_size, dtype=np.float32)
        cut[:cut_position+1] = 1.0
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

    ego_graphs_train, ego_graphs_valid, ego_graphs_test = torch.LongTensor(ego_graphs_train), torch.LongTensor(ego_graphs_valid), torch.LongTensor(ego_graphs_test)
    cut_train, cut_valid, cut_test = torch.FloatTensor(cut_train), torch.FloatTensor(cut_valid), torch.FloatTensor(cut_test)

    pe = None
    if args.pe_type == 1:
        pe = torch.load(f'data/{args.dataset}-embedding-{args.hidden_size}.pt')
    elif args.pe_type == 2:
        pe = np.fromfile("data/paper100m.pro", dtype=np.float32).reshape(-1, 128)
        pe = torch.FloatTensor(pe)
        if args.hidden_size < 128:
            pe = pe[:, :args.hidden_size]

    data = dataset[0]
    if len(data.y.shape) == 1:
        data.y = data.y.unsqueeze(1)
    adj = None
    if args.mask:
        adj = torch.BoolTensor(~np.load(f'data/{args.dataset}-ego-graphs-adj-{args.ego_size}.npy'))

    num_classes = dataset.num_classes

    train_dataset = NodeClassificationDataset(data.x, data.y, ego_graphs_train, pe, args, num_classes, adj, cut_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=batcher(train_dataset), pin_memory=True)

    valid_dataset = NodeClassificationDataset(data.x, data.y, ego_graphs_valid, pe, args, num_classes, adj, cut_valid)
    valid_loader = DataLoader(valid_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=batcher(valid_dataset), pin_memory=True)

    test_dataset = NodeClassificationDataset(data.x, data.y, ego_graphs_test, pe, args, num_classes, adj, cut_test)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=batcher(test_dataset), pin_memory=True)

    model = TransformerModel(data.x.size(1)+1, args.hidden_size,
                             args.num_heads, args.hidden_size,
                             args.num_layers, num_classes, 
                             args.input_dropout, args.hidden_dropout,
                             layer_norm=args.layer_norm, 
                             src_scale=args.src_scale).to(device)
    wandb.watch(model, log='all')

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model parameters:', pytorch_total_params)

    if not os.path.exists('saved'):
        os.mkdir('saved')

    if torch.cuda.device_count() > 1:
        model.module.init_weights()
    else:
        model.init_weights()

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
        if args.scheduler == 'noam':
            optimizer = NoamOptim(optimizer, args.hidden_size if args.hidden_size > 0 else data.x.size(1), n_warmup_steps=args.warmup) #, init_lr=args.lr)
        elif args.scheduler == 'linear':
            optimizer = LinearOptim(optimizer, n_warmup_steps=args.warmup, n_training_steps=args.epochs * len(train_loader), init_lr=args.lr)

    for epoch in range(1, 1 + args.epochs):
        # lp = LineProfiler()
        # lp_wrapper = lp(train)
        # loss = lp_wrapper(model, train_loader, device, optimizer, args)
        # lp.print_stats()
        loss = train(model, train_loader, device, optimizer, args)

        train_output = valid_output = test_output = ''
        if epoch >= 10 and epoch % args.log_steps == 0:
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
                    if torch.cuda.device_count() > 1:
                        torch.save(model.module.state_dict(), 'saved/' + exp_name + '.pt')
                    else:
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
