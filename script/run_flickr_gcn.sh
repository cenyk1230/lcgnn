cuda=$1
echo 'using cuda' $cuda
seeds=$(seq 0 9)
for i in $seeds
do
    echo 'running for seed' $i
    CUDA_VISIBLE_DEVICES=$cuda python train_gnn.py --dataset flickr --model gcn --num_layers 5 --ego_size 128 --hidden_size 128 --input_dropout 0.0 --hidden_dropout 0.3 --lr 0.01 --weight_decay 0.0005 --seed $i
done