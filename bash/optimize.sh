opt_log_dir=log/optimize/
if [ ! -d $opt_log_dir ]; then
    mkdir -p $opt_log_dir
fi
log_file=$opt_log_dir/opt.log

visdom_port=8201
display_freq=4096
port_offset=5000
dist_port=$(expr $visdom_port - $port_offset)
batch_size=512
strategy=opt_default

data_root=./ihmr_data/
model_root=$data_root'models/' # pretrain weights and MANO weights
blur_kernel_dir=$model_root'blur_kernel/' # for motion blur augmentation
param_root=$data_root'hand26m/param' # directory to store the parameters including joints and parameter

hand26m_anno_path=hand26m/annotation/test_inter_close.pkl 
hand26m_pred_path=hand26m/prediction/test_inter_close.pkl

CUDA_VISIBLE_DEVICES=0,1 python3 -u -m torch.distributed.launch \
    --nproc_per_node=2 --master_port=$dist_port src/optimize.py --dist \
    --checkpoints_dir checkpoints/optimize \
    --model_type opt \
    --batchSize $batch_size \
    --data_root $data_root \
    --param_root $param_root \
    --model_root $model_root \
    --hand26m_anno_path $hand26m_anno_path \
    --hand26m_pred_path $hand26m_pred_path \
    --opt_dataset hand26m \
    --save_mid_freq 10 \
    --strategy $strategy \
    --optimizer adam \
    --display_port $visdom_port  2>&1 | tee $log_file