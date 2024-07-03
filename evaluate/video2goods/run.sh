output_dir=$1
name=$2
gt_file=$3

python 1-convert_feat.py \
   $output_dir/photo/query.feat \
   $output_dir/goods/doc.feat \
   512 \
   res/cross_domain_$name

python 2-create_index.py \
   res/cross_domain_$name

python 3-search.py \
   res/cross_domain_$name \
   1000

python 4-analysis.py \
   res/cross_domain_$name \
   $output_dir/photo/query.feat \
   $output_dir/goods/doc.feat \
   $gt_file

python 6-vis.py main \
  res/cross_domain_$name/search_res.npy \
  res/cross_domain_$name/search_res_dis.npy \
  $output_dir/photo/query.feat \
  $output_dir/goods/doc.feat \
  $gt_file \
  res/cross_domain_$name/vis.html
