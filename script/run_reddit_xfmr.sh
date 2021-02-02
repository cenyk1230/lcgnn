cuda=$1
echo 'using cuda' $cuda
seeds=$(seq 1 4)
for i in $seeds
do
    echo 'running for seed' $i
    CUDA_VISIBLE_DEVICES=$cuda python train.py --dataset reddit --num_layers 5 --num_heads 4 --ego_size 256 --input_dropout 0.3 --hidden_dropout 0.4 --weight_decay 0.005 --layer_norm 0 --early_stopping 50 --seed $i
done