import argparse
import os
from best_configs import configs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sensitivity')
    parser.add_argument('--cuda', type=str, default=0)
    parser.add_argument('--model', type=str, default='transformer')
    parser.add_argument('--dataset', type=str, default='flickr')
    parser.add_argument('--ego_size', default=[64], type=int, nargs='+')
    args = parser.parse_args()
    print(args)

    # all_models = ["transformer", "gcn", "sage", "gat"]
    # all_datasets = ["flickr", "reddit", "yelp", "amazon"]

    if args.model == "transformer":
        command = f"CUDA_VISIBLE_DEVICES={args.cuda} python train.py --project lcgnn-param"
    else:
        command = f"CUDA_VISIBLE_DEVICES={args.cuda} python train_gnn.py --model {args.model} --project lcgnn-param"
    command += f" --dataset {args.dataset} "

    config = configs[args.model][args.dataset]
    num = 1
    if args.dataset == "flickr":
        num = 5
        config["early_stopping"] = 50
    elif args.dataset == 'reddit':
        num = 2
        config["early_stopping"] = 40
    elif args.dataset == 'yelp':
        num = 2
        config["early_stopping"] = 30
    elif args.dataset == 'amazon':
        num = 1
        config["early_stopping"] = 20

    for seed in range(num):
        for ego_size in args.ego_size:
            cur_command = command + " ".join([f"--{key} {value}" if key != 'ego_size' else f"--{key} {ego_size}" for key, value in config.items()]) + f" --seed {seed}"
            print(cur_command)

            os.system(cur_command)
