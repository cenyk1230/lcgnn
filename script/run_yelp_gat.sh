cuda=$1
echo 'using cuda' $cuda
seeds=$(seq 1 2)
for i in $seeds
do
    echo 'running for seed' $i
    CUDA_VISIBLE_DEVICES=$cuda python train_gnn.py --dataset yelp --model gat --num_layers 4 --num_heads 4 --ego_size 128 --hidden_size 1024 --input_dropout 0.0 --hidden_dropout 0.2 --lr 0.001 --weight_decay 0.0005 --seed $i
done