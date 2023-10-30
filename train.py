import os
import torch
import numpy as np
import yaml
from utils_new.dist_util import *
from utils_new.util import *
from tqdm import tqdm
from torch.utils import data
from torch import nn, optim
from dataset.base_dataset import *
from dataset.image_datasets import load_data
import torchvision
from einops import rearrange, repeat
from matplotlib import pyplot as plt
# from eval.fid_evaluator import FIDEvaluator
# from eval.prec_and_recall_evaluator import PrecAndRecallEvaluator
from torch.utils.tensorboard import SummaryWriter
from utils_new.eval_util import flatten_results_dict
import argparse

class Trainer:

    def __init__(
        self,
        model,
        diffusion,
        ema,
        loader,
        device,
        optim,
        distributed,
        work_dir,
        iterations,
        log_image_interval,
        save_ckpt_interval,
        max_images,
        evaluator,
        eval_interval,
    ):

        self.device = device
        self.work_dir = work_dir
        self.make_work_dir()

        if get_rank()==0:
            self.writer = SummaryWriter(self.work_dir)
        else:
            self.writer = None

        self.iterations = iterations
        self.iter = 0
        pbar = range(int(iterations) + 1)

        if get_rank() == 0:
            self.pbar = tqdm(pbar, initial=0, dynamic_ncols=True, smoothing=0.01)
        else:
            self.pbar = pbar

        self.data = loader
        self.optim = optim

        self.diffusion = diffusion
        self.model = model
        self.ema = ema
        self.distributed = distributed

        if distributed:
            self.m_module = self.model.module
        else:
            self.m_module = self.model

        self.log_image_interval = log_image_interval
        self.save_ckpt_interval = save_ckpt_interval
        self.max_images = max_images

        self.evaluator = evaluator
        self.eval_interval = eval_interval

        self.kwargs = {}
        self.log_schedule()
            
    def train(self, args):
        for idx in self.pbar:
            self.iter = idx
            if self.iter > self.iterations:
                print("Done!")
                break

            requires_grad(self.model, True)

            x_start, noise = self.on_train_epoch_start(args)

            t = torch.randint(
                0,
                self.diffusion.num_timesteps,
                (x_start.shape[0],),
                device=self.device,
            ).long()
            #print(args.seperate_channel)
            loss, loss_dict = self.diffusion.training_losses(
                    self.model, 
                    x_start=x_start, 
                    t=t, device=self.device, 
                    noise=noise, 
                    seperate_channel_loss=args.seperate_channel_loss
                    )

            loss_reduced = reduce_loss_dict(loss_dict)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            accumulate(self.ema, self.m_module)
            self.log_metric(loss_reduced)

            self.on_train_epoch_end()

    def on_train_epoch_start(self, args):

        batch, mask = next(iter(self.data))

        ##### comment out # for swapping classes index to channel-wise 255 #####
        print(args.num_defect)
        unique_values = range(0, args.num_defect+1)  # Channel fusion 

        num_classes = len(unique_values)

        # Create an empty multi-channel image
        b, _, h, w = mask.shape

        onehot_mask = torch.zeros(b, num_classes, h, w)
        onehot_mask.scatter_(1, mask.long(), 1)
        mask = mask.squeeze(1)
       
        img_mask = torch.cat((batch, onehot_mask), dim=1)

        img_mask = img_mask.to(self.device)
        noise = None

        if self.iter == 0:
            self.kwargs['noise'] = noise[:self.max_images,:,:] if noise is not None else None
            self.kwargs['shape'] = [self.max_images, *img_mask.shape[1:]]
            self.kwargs["num_timesteps"] = None
        return img_mask, noise

    def on_train_epoch_end(self):
        if (self.iter) % self.log_image_interval == 0:
            self.log_images()

        if (self.iter) % self.save_ckpt_interval == 0:
            self.save_ckpt()

        # if (self.iter) % self.eval_interval == 0:
        #     self.eval()

    def log_metric(self, dict):
        if get_rank() == 0:            
            self.pbar.set_description(
                (
                    ' '.join([f"{k}: {v.mean().item():.4f}"  for k,v in dict.items()])
                )
            )
            for k, v in dict.items():
                self.writer.add_scalar(f'train/{k}', (v).mean(), self.iter)

    def semantic_mask_to_rgb(self, mask):
        # Define a color for each of the 11 possible class values (0 through 10)
        colors = [
            (0, 0, 0),       # 0: Black
            (0, 0, 255),     # 1: Blue
            (0, 255, 0),     # 2: Green
            (255, 0, 0),     # 3: Red
            (0, 255, 255),   # 4: Yellow
            (255, 0, 255),   # 5: Magenta
            (255, 255, 0),   # 6: Cyan
            (128, 0, 0),     # 7: Dark Red
            (0, 128, 0),     # 8: Dark Green
            (0, 0, 128),     # 9: Dark Blue
            (128, 128, 128)  # 10: Gray
        ]

        # Convert the grayscale mask to an RGB image
        h, w = mask.shape
        rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(11):  # we have 11 classes
            rgb_mask[mask == i] = colors[i]
        return rgb_mask     
    
    def argmax_above_threshold(self, softmax_output, threshold):
        # Compute the argmax values along the depth (1-axis)
        argmax_depth = torch.argmax(softmax_output, dim=1) + 1  # Adding 1 to make it 1-based
        
        # Compute the max values along the depth (1-axis)
        max_values = torch.max(softmax_output, dim=1).values  # .values to get the actual max values

        # Set positions with max values below the threshold to 0
        argmax_depth[max_values < threshold] = 0
        
        return argmax_depth
    
    def log_images(self, img_name=None):
        self.model.eval()
        if img_name is None:
            img_name = str(self.iter).zfill(6)
        model_kwargs = {}
        
        images, intermediates = self.diffusion.p_sample_loop(
                model=self.model, 
                shape=self.kwargs['shape'],
                progress=True if get_rank()==0 else False,
                noise=self.kwargs['noise'],
                return_intermediates=True,
                model_kwargs=self.kwargs,
                log_interval=self.diffusion.num_timesteps // 10
                )
        
        gathered_images = all_gather(images)
        gathered_img = torch.cat(gathered_images, dim=0)[:, :3, :, :]
        torch.set_printoptions(profile="full")

        
        #calculate the final masks
        gathered_masks = torch.cat(gathered_images, dim=0)[:, 3:, :, :]
        softmax_output = F.softmax(gathered_masks, dim=1)
        argmax_depth = torch.argmax(softmax_output, dim=1)

        batch, _, _ = argmax_depth.shape

        if get_rank() == 0:

            torchvision.utils.save_image(
                gathered_img, 
                f'{self.sample_dir}samples_img_{img_name}.png',
                normalize=True, range=(-1, 1), nrow=self.max_images
                )
            
            for i in range(batch):
                rgb_mask = self.semantic_mask_to_rgb(argmax_depth[i].cpu().numpy())
                #im_masks = Image.fromarray(final_masks[i].cpu().numpy())
                im = Image.fromarray(rgb_mask)
                im.save(f'{self.sample_mask_dir}samples_mask_{img_name}_{i}.png')
  
        self.model.train()
        synchronize()

    def log_schedule(self):
        if get_rank() == 0:
            # schedule
            plt.figure(figsize=(5, 10))
            plt.subplot(211)
            plt.plot([i for i in range(self.diffusion.num_timesteps)], self.diffusion.betas.cpu(), label='betas')
            plt.title("schedule: {}".format(self.diffusion.schedule))
            plt.xlabel('t')
            plt.ylabel('betas')

            plt.subplot(212)
            plt.plot([i for i in range(self.diffusion.num_timesteps)], self.diffusion.alphas_cumprod.cpu(), label='alphas_cumprod')
            plt.xlabel('t')
            plt.ylabel('alpha_cumprod')

            plt.savefig('{}/schedule.png'.format(self.work_dir))

    def save_ckpt(self):
        if get_rank() == 0:
            torch.save(
                {
                    "model": self.m_module.state_dict(),
                    "optimizer": self.optim.state_dict(),
                    "ema": self.ema.state_dict(),
                },
                f"{self.checkpoint_dir}diffusion_{str(self.iter).zfill(6)}.pt"
            )
    
    def make_work_dir(self):
        
        self.sample_dir = os.path.join(self.work_dir, 'sample/')
        self.sample_mask_dir = os.path.join(self.sample_dir, 'mask/')
        self.checkpoint_dir= os.path.join(self.work_dir,'checkpoint/')

        if get_rank() == 0:

            os.makedirs(os.path.dirname(self.sample_dir), exist_ok=True)
            os.makedirs(os.path.dirname(self.sample_mask_dir), exist_ok=True)
            os.makedirs(os.path.dirname(self.checkpoint_dir), exist_ok=True)

    def eval(self):
        if get_rank()==0:
            result = {self.evaluator.metrics: self.evaluator.eval(self.model, self.kwargs['shape'])}
            # tensor board
            result = flatten_results_dict(result)
            print(result)
            for k, v in result.items():
                self.writer.add_scalar(f'eval/{k}', v, self.iter)

def main():
    device = "cuda"
    # parse necessary information
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='/')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--work_dir", type=str)
    parser.add_argument("--seperate_channel_loss", type=int, default=0)
    parser.add_argument("--num_defect", type=int, default=5)
    args = parser.parse_args()

    # read config
    f = open(args.config, 'r', encoding='utf-8')
    d = yaml.safe_load(f)

    # dump config
    os.makedirs(os.path.dirname(args.work_dir), exist_ok=True)
    config_path = os.path.join(args.work_dir, 'config_dump.yml')
    #save_dict_to_yaml(d, config_path)


    # distribute training
    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = n_gpu > 1
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    
    # prepare model and diffusion
    diffusion = instantiate_from_config(d['diffusion']).to(device)
    model = instantiate_from_config(d['model']).to(device)

    model_ema = instantiate_from_config(d['model']).to(device)
    model_ema.eval()
    accumulate(model_ema, model, 0)

    optimizer = optim.AdamW(
            list(model.parameters()), lr=d['optimizer']['params']['lr'], weight_decay=d['optimizer']['params']['weight_decay']
        )
    
    if 'ckpt' in d['model'].keys() and d['model']['ckpt'] is not None:
        print("load model:", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        model.load_state_dict(ckpt["model"])
        model_ema.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt['optimizer'])


    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )

    #dataset = instantiate_from_config(d['data'])


    dataset = load_data(
            data_dir=d['data']['params']['dir'],
            batch_size=d['data']['bs_per_gpu'],
            image_size=d['data']['params']['resolution'],
            num_images=d['data']['params']['num_image_train']
    )



    # start training
    trainer = Trainer(
        model = model,
        diffusion = diffusion,
        ema = model_ema,
        loader=dataset,
        optim = optimizer,
        device=device,
        distributed=distributed,
        work_dir=args.work_dir,
        evaluator=None,
        **d['train']
    )
    trainer.train(args)


if __name__ == "__main__":
    main()
