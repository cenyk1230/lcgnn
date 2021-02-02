cuda=$1
echo 'using cuda' $cuda
seeds=$(seq 1 2)
for i in $seeds
do
    echo 'running for seed' $i
    CUDA_VISIBLE_DEVICES=$cuda python train.py --dataset yelp --num_layers 3 --num_heads 1 --ego_size 128 --hidden_size 256 --input_dropout 0.1 --hidden_dropout 0.3 --weight_decay 0.0005 --layer_norm 0 --early_stopping 50 --seed $i
done