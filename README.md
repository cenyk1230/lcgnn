# LCGNN

## Dependency

The code is written in Python 3. We use PyTorch 1.5.1 with CUDA 10.1 to train our models. 

The dependencies of our code are:
* torch (https://pytorch.org/get-started/locally/)
* torch_geometric (https://github.com/rusty1s/pytorch_geometric)
* dgl (https://github.com/dmlc/dgl)
* ogb (https://github.com/snap-stanford/ogb)
* wandb (https://github.com/wandb/client)
* localgraphclustering (https://github.com/kfoynt/LocalGraphClustering)

Please follow the instructions to install these dependencies.

## Usage

### Preprocessing

Use the following command to run local clustering for all nodes and save the data:

```
python preprocess.py --task {task} --dataset {name} --ego_size {ego_size}
```

The preprocessed data will be saved into `./data/{name}-lc-ego-graphs-{ego_size}.npy` and `./data/{name}-lc-conds-{ego_size}.npy` where task is chosen from [`saint`, `ogbn`], name is chosen from [`flickr`, `reddit`, `yelp`, `amazon`, `products`, `papers100M`], ego_size denotes the maximum cluster size. 

### Training

Our code supports four inductive datasets including flickr, reddit, yelp, amazon and two OGB datasets including products, papers100M.

To run LCGNN-Transformer, use:

```
python train.py --dataset flickr|reddit|yelp|amazon|products|papers100M
```

To run LCGNN-GCN/GAT/SAGE, use:
```
python train_gnn.py --dataset flickr|reddit|yelp|amazon|products --model gcn/gat/sage
```

You can find the best configuration in the `best_configs.py`. 

### Datasets
* Four inductive datasets can be downloaded from [here](https://drive.google.com/drive/folders/1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz).
* OGB datasets will be automatically downloaded through OGB APIs. You can find more details about OGB datasets from [here](https://ogb.stanford.edu/docs/home/).
