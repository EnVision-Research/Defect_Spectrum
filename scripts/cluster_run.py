import os
import random
gpu = 1
arch = 'ampere'
path = os.getcwd()
excludes= "node33"
image = 'harbor.smoa.cc/public/smore_quant:vdatacode.rc2.cu11'
object_list = ['bottle', 'carpet', 'grid', 'leather', 'screw', 'tile', 'transistor']
defect_num_list = [3, 5, 5, 5, 5, 5, 4]


def replace_bottle_with_object(input_file, output_file, object, defect_num):
    with open(input_file, 'r') as f:
        content = f.read()

    content = content.replace('bottle', object)
    content = content.replace('--num_defect 5', '--num_defect {}'.format(defect_num))
    content = content.replace('in_channels: 9', 'in_channels: {}'.format(defect_num+4))
    content = content.replace('out_channels: 9', 'out_channels: {}'.format(defect_num+4))
    with open(output_file, 'w') as f:
        f.write(content)

for idx, object in enumerate(object_list):
    name = 'diffgen-{}'.format(object) + '-' + str(random.randint(0, 10000))
    replace_bottle_with_object("/dataset/shuaiyang/research/industry_seg/unified_diffusion_diff_aug/config/large_recep.yml",
                               "/dataset/shuaiyang/research/industry_seg/unified_diffusion_diff_aug/config/large_recep_{}.yml".format(object),
                               object,
                               defect_num_list[idx])
    replace_bottle_with_object("../train_small_6ch_mse.sh",
                               "./train_small_6ch_mse_{}.sh".format(object),
                               object,
                               defect_num_list[idx])

    cmd = 'source /dataset/shuaiyang/.bashrc && conda activate diff_aug && cd /dataset/shuaiyang/research/industry_seg/unified_diffusion_diff_aug/scripts && bash train_large_6ch_mse_{}.sh'.format(object)
    print(cmd)
    os.system('~/schedctl create --arch {} --image {} --excludes {} --name {} --gpu {} --cmd \"{}\"'.format(arch, image, excludes, name, gpu, cmd))
    # break
