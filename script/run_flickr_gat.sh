cuda=$1
echo 'using cuda' $cuda
seeds=$(seq 0 9)
for i in $seeds
do
    echo 'running for seed' $i
    CUDA_VISIBLE_DEVICES=$cuda python train_gnn.py --dataset flickr --model gat --num_layers 5 --num_heads 4 --ego_size 128 --hidden_size 512 --input_dropout 0.2 --hidden_dropout 0.1 --lr 0.001 --weight_decay 0.0 --seed $i
done