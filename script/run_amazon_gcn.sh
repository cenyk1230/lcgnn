cuda=$1
echo 'using cuda' $cuda
seeds=$(seq 1 2)
for i in $seeds
do
    echo 'running for seed' $i
    CUDA_VISIBLE_DEVICES=$cuda python train_gnn.py --dataset amazon --model gcn --num_layers 4 --num_heads 1 --ego_size 128 --hidden_size 512 --input_dropout 0.0 --hidden_dropout 0.2 --weight_decay 0.0 --seed $i
done