train_log_dir=log/baseline/train_logs/
if [ ! -d $train_log_dir ]; then
    mkdir -p $train_log_dir
fi
curr_date=$(date +'%m_%d_%H_%M') 
log_file=$train_log_dir$curr_date".log"

visdom_port=8201
display_freq=4096
port_offset=5000
dist_port=$(expr $visdom_port - $port_offset)

data_root=./ihmr_data/
model_root=$data_root'models/' # pretrain weights and MANO weights
blur_kernel_dir=$model_root'blur_kernel/' # for motion blur augmentation
param_root=$data_root'hand26m/param' # directory to store the parameters including joints and parameters

hand26m_anno_path=hand26m/annotation/train.pkl
pretrain_weights=$model_root'pretrain/baseline.pth'

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -u -m torch.distributed.launch \
    --nproc_per_node=4 --master_port=$dist_port src/train_baseline.py --dist \
    --display_freq $display_freq \
    --checkpoints_dir checkpoints/baseline \
    --pretrain_weights $pretrain_weights \
    --model_type baseline \
    --lr 1e-4 \
    --lr_decay_type cosine \
    --total_epoch 20 \
    --data_root $data_root \
    --param_root $param_root \
    --model_root $model_root \
    --hand26m_anno_path $hand26m_anno_path \
    --train_datasets hand26m \
    --use_random_flip \
    --use_random_rescale \
    --use_random_position \
    --use_random_rotation \
    --use_color_jittering \
    --use_motion_blur \
    --blur_kernel_dir $blur_kernel_dir \
    --display_port $visdom_port  2>&1 | tee $log_file
