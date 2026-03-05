import os

import torch

import pandas as pd

from mmengine.config import Config

from mmengine.runner import Runner

from mmdet3d.utils import register_all_modules

from mmdet3d.apis import init_model



register_all_modules()



# --- PATHS ---

mmdet_path = '/home/sb2ek/curriculum/mmdetection3d'

config_file = os.path.join(mmdet_path, 'configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d.py')

checkpoint_file = 'checkpoints/fcos3d_baseline.pth'

data_root = 'data/nuscenes/' 

ann_file = 'nuscenes_infos_train.pkl' 




cfg = Config.fromfile(config_file)

cfg.work_dir = './work_dirs/loss_analysis'

cfg.val_dataloader.dataset.data_root = data_root

cfg.val_dataloader.dataset.ann_file = ann_file

cfg.val_dataloader.dataset.pipeline = eval_loss_pipeline 



# 4. Initialize Runner & Model

runner = Runner.from_cfg(cfg)

dataloader = runner.val_dataloader

model = init_model(config_file, checkpoint_file, device='cuda:0')



results_list = []



print(f"Calculating losses for {ann_file}...")



for i, data_batch in enumerate(dataloader):

    # 'training=True' ensures the preprocessor maps GT to the loss head

    data_batch = model.data_preprocessor(data_batch, training=True)

    

    with torch.no_grad():

        try:

            # Running in 'loss' mode to get the breakdown

            losses = model.forward(data_batch['inputs'], data_batch['data_samples'], mode='loss')

            

            for idx, sample in enumerate(data_batch['data_samples']):

                filename = os.path.basename(sample.img_path)

                

                row = {

                    'filename': filename,

                    'loss_cls': losses.get('loss_cls', torch.tensor(0)).item(),

                    'loss_bbox': losses.get('loss_bbox', torch.tensor(0)).item(),

                    'loss_offset': losses.get('loss_offset', torch.tensor(0)).item(),

                    'total_loss': sum(l.item() for l in losses.values() if isinstance(l, torch.Tensor))

                }

                results_list.append(row)

        except Exception as e:

            print(f"Error on batch {i}: {e}")



    if i % 20 == 0:

        print(f"Processed {i} samples...")



# 5. Save

df = pd.DataFrame(results_list)

df.to_csv('fcos3d_val_losses.csv', index=False)

print("Finished. Results in fcos3d_val_losses.csv")
