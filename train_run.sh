export OMP_NUM_THREADS=4

# TODO 1. The save path for trained framework.
checkpoint_path="./Dataset/checkpoints/checkpoint.pth.tar"

# TODO 2. The path to the pre-trained model.
# Due to the restriction imposed by Kuaishou on code sharing, which prevents us from making the pre-training framework code public.
# However, we will open-source the pre-trained model weights and provide a link to access them on Baidu Netdisk.
pretrained_path="./Dataset/checkpoints/pretrain.pth.tar"

mkdir -p ./outputs/logs/

python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="11.37.6.163" --master_port=11115 main_cross_domain_emb.py \
    --embedding_size 512 \
    --workers 10 \
    --epochs 10 \
    --decay_epochs 3 \
    --batch_size 80 \
    --lr 0.01 \
    --save_freq 50000 \
    --checkpoints $checkpoint_path \
    --pretrained $pretrained_path \
    --mixed_precision_training \
    --finetune \
    --train_file ./Dataset/Real400K/train_file/new_train_98000.txt \
    --cls_file  ./Dataset/Real400K/train_file/cluster_10.txt\
    --cls_num 10 \
    --sample_info_file ./Dataset/Real400K/train_file/new_train_98000.txt\
    --goods_img_root ./Dataset/Real400K/goods/images \
    --goods_text_root ./Dataset/Real400K/goods/text  \
    --photo_img_root ./Dataset/Real400K/video/images \
    --photo_text_root ./Dataset/Real400K/video/images \
    --clip_length 5 \
    >> ./outputs/logs/training.log
