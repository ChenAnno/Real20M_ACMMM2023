export OMP_NUM_THREADS=4

# TODO 1. The save path for trained framework.
model_path="./Dataset/checkpoints/checkpoint.pth.tar"
# TODO 2. The version number of the model, for convenient storage of visualization files and result files
version=video2goods-0910
# TODO 3. Save path for visualization files and result files.
output_path="./evaluate/feat/$version"
# TODO 4. Path for the gt file, better to set the absolute path for pretrained pth. This file needs to be obtained through application.
gt_path="XXX/Datset/test_file/eval_gt_video2goods.txt"

if [ -e $output_path ]
then
    rm -rf $output_path
fi

mkdir -p $output_path/goods
mkdir -p $output_path/photo

# micro-video features
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=52293 main_cross_domain_emb.py \
    --embedding_size 512 \
    --workers 10 \
    --epochs 10000 \
    --decay_epochs 30 \
    --batch_size 80 \
    --lr 0.005 \
    --save_freq 50000 \
    --finetune \
    --mixed_precision_training \
    --goods_img_root ./Dataset/Real400K/goods/images \
    --goods_text_root ./Dataset/Real400K/goods/text  \
    --photo_img_root ./Dataset/Real400K/video/images \
    --photo_text_root ./Dataset/Real400K/video/images \
    --clip_length 5 \
    --test_file './Dataset/test_file/photo_query.txt' \
    --evaluate \
    --resume $model_path \
    --output_dir $output_path/photo

# goods features
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=52293 main_cross_domain_emb.py \
    --embedding_size 512 \
    --workers 10 \
    --epochs 10000 \
    --decay_epochs 30 \
    --batch_size 80 \
    --lr 0.005 \
    --save_freq 50000 \
    --finetune \
    --mixed_precision_training \
    --goods_img_root ./Dataset/Real400K/goods/images \
    --goods_text_root ./Dataset/Real400K/goods/text  \
    --photo_img_root ./Dataset/Real400K/video/images \
    --photo_text_root ./Dataset/Real400K/video/images \
    --clip_length 5 \
    --test_file './Dataset/Real400K/test_file/all_goods_177839.txt' \
    --evaluate \
    --resume $model_path \
    --output_dir $output_path/goods

cat $output_path/photo/* > $output_path/photo/query.feat
cat $output_path/goods/* > $output_path/goods/doc.feat

# compute the metriecs and generate visualization files
cd evaluate/video2goods
sh run.sh $output_path $version $gt_path
