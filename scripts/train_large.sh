#!/bin/sh
CUDA_VISIBLE_DEVICES="0" \
python -m torch.distributed.launch \
--nproc_per_node=1 \
--master_port=29513 \
train.py \
--work_dir /home/zfchen/working/diff_aug/COTTON_dataset/large_recep \
--config /home/zfchen/working/diff_aug/config/small_recep.yml \
--seperate_channel 0 \
--num_defect 2
