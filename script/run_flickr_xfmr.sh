cuda=$1
echo 'using cuda' $cuda
seeds=$(seq 0 9)
for i in $seeds
do
    echo 'running for seed' $i
    CUDA_VISIBLE_DEVICES=$cuda python train.py --dataset flickr --num_layers 5 --num_heads 4 --ego_size 256 --hidden_size 64 --input_dropout 0.4 --hidden_dropout 0.3 --weight_decay 0.0005 --layer_norm 1 --warmup 4000 --early_stopping 50 --seed $i
done