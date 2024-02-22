#!/bin/sh
CUDA_VISIBLE_DEVICES="7" \
python -m torch.distributed.launch \
--nproc_per_node=1 \
--master_port=29516 \
train.py \
--config /home/zfchen/working/diff_aug/config/small_recep.yml \
--work_dir /home/zfchen/working/diff_aug/COTTON_dataset/large_recep \
--seperate_channel_loss 0 \
--num_defect 2