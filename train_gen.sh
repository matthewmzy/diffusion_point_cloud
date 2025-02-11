CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
    --master_addr 127.0.0.1 \
    --master_port 29500 \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    train_gen.py \
    --lr 5e-4 \
    --point_dim 4 \
    --sample_num_points 256 \
