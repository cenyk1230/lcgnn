cuda=$1
echo 'using cuda' $cuda
seeds=$(seq 1 2)
for i in $seeds
do
    echo 'running for seed' $i
    CUDA_VISIBLE_DEVICES=$cuda python train_gnn.py --dataset amazon --model gat --num_layers 5 --num_heads 4 --ego_size 64 --hidden_size 1024 --input_dropout 0.2 --hidden_dropout 0.1 --lr 0.001 --weight_decay 0.0 --seed $i
done