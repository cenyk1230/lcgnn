cuda=$1
echo 'using cuda' $cuda
seeds=$(seq 1 4)
for i in $seeds
do
    echo 'running for seed' $i
    CUDA_VISIBLE_DEVICES=$cuda python train_gnn.py --dataset products --model gcn --num_layers 4 --num_heads 1 --ego_size 128 --hidden_size 256 --input_dropout 0.2 --hidden_dropout 0.4 --lr 0.001 --weight_decay 0.0 --seed $i
done