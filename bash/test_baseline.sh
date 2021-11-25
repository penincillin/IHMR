test_log_dir=log/baseline/test_logs/
if [ ! -d $test_log_dir ]; then
    mkdir -p $test_log_dir
fi
batch_size=512

visdom_port=8201
display_freq=4096
port_offset=5000
dist_port=$(expr $visdom_port - $port_offset)

data_root=./ihmr_data/
model_root=$data_root'models/' # pretrain weights and MANO weights
blur_kernel_dir=$model_root'blur_kernel/' # for motion blur augmentation
param_root=$data_root'hand26m/param' # directory to store the parameters including joints and parameters
hand26m_anno_path=hand26m/annotation/test_inter_close.pkl 

test_epoch=pretrain
test_dataset=hand26m

log_file=$test_log_dir$test_dataset".log"
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -u -m torch.distributed.launch  \
    --nproc_per_node=4 --master_port=$dist_port src/test_baseline.py --dist \
    --checkpoints_dir checkpoints/baseline \
    --model_type baseline \
    --test_epoch $test_epoch \
    --data_root $data_root \
    --param_root $param_root \
    --model_root $model_root \
    --batchSize $batch_size \
    --hand26m_anno_path $hand26m_anno_path \
    --test_dataset $test_dataset \
    2>&1 | tee $log_file