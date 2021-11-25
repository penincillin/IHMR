train_log_dir=log/mlp/train_logs/
if [ ! -d $train_log_dir ]; then
    mkdir -p $train_log_dir
fi
curr_date=$(date +'%m_%d_%H_%M') 
log_file=$train_log_dir$curr_date".log"

batch_size=512
visdom_port=8202
display_freq=4096
port_offset=5000
dist_port=$(expr $visdom_port - $port_offset)
strategy=mlp_default

data_root=./ihmr_data/
model_root=$data_root'models/' # pretrain weights and MANO weights
blur_kernel_dir=$model_root'blur_kernel/' # for motion blur augmentation
param_root=$data_root'hand26m/param' # directory to store the parameters including joints and parameters

hand26m_anno_path=hand26m/annotation/train_inter_close.pkl
hand26m_pred_path=hand26m/prediction/train_inter_close.pkl
pretrain_weights_dir=$model_root'pretrain/mlp'

CUDA_VISIBLE_DEVICES=0,1 python3 -u -m torch.distributed.launch \
    --nproc_per_node=2 --master_port=$dist_port src/train_mlp.py --dist \
    --model_type mlp \
    --checkpoints_dir checkpoints/mlp \
    --pretrain_weights_dir $pretrain_weights_dir \
    --display_freq $display_freq \
    --batchSize $batch_size \
    --data_root $data_root \
    --param_root $param_root \
    --model_root $model_root \
    --hand26m_anno_path $hand26m_anno_path \
    --hand26m_pred_path $hand26m_pred_path \
    --train_datasets hand26m \
    --strategy $strategy \
    --use_opt_params \
    --display_port $visdom_port  2>&1 | tee $log_file