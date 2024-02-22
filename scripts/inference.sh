#!/bin/sh
CUDA_VISIBLE_DEVICES='2, 3' \
python -m torch.distributed.launch \
--nproc_per_node=2 \
--master_port=29511 \
inference_for_perception.py \
--step_inference 400 \
--sample_dir '/home/zfchen/working/diff_aug/generation_for_perception_test/screw_exp4' \
--large_recep '/home/zfchen/working/diff_aug/VISION_dataset/screw_large_recep/checkpoint/diffusion_150000.pt' \
--small_recep '' \
--num_defect 3 \
--large_recep_config '/home/zfchen/working/diff_aug/config/large_recep_vision_screw.yml' \
--small_recep_config '/home/zfchen/working/diff_aug/config/small_recep.yml' \
