cuda=$1
echo 'using cuda' $cuda
seeds=$(seq 1 2)
for i in $seeds
do
    echo 'running for seed' $i
    CUDA_VISIBLE_DEVICES=$cuda python train.py --dataset amazon --num_layers 5 --num_heads 4 --ego_size 64 --hidden_size 128 --input_dropout 0.1 --hidden_dropout 0.2 --weight_decay 0.0 --layer_norm 1 --early_stopping 50 --seed $i
done