from mmdet3d.registry import TRANSFORMS, HOOKS
from mmcv.transforms import BaseTransform
from mmcv.transforms import LoadImageFromFile

from utils.add_fog import add_fog_beta, estimate_airlight
from mmengine.hooks import Hook
from mmengine.fileio import get
from mmengine.logging import print_log
import logging
import mmcv
import numpy as np
import os
from typing import List, Optional

@TRANSFORMS.register_module()
class LoadSingleImageWithDepth(LoadImageFromFile):
    def __init__(self,
                depth_root: str,
                to_float32: bool = False,
                color_type: str = 'unchanged',
                imdecode_backend: str = 'cv2',
                sweeps_num: int = 10,
                load_dim: int = 5,
                use_dim: List[int] = [0, 1, 2, 4],
                backend_args: Optional[dict] = None,
                pad_empty_sweeps: bool = False,
                remove_close: bool = False,
                test_mode: bool = False) -> None:

        super().__init__(
            to_float32=to_float32,
            color_type=color_type,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args,
        )

        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        self.use_dim = use_dim
        self.backend_args = backend_args
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        self.cam = 'CAM_FRONT'
        self.depth_root = depth_root
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend

    def transform(self, results: dict) -> dict:
        #results["img_info"] contain info of sample idx.

        print(results['images'])
        camera_type = list(results['images'].keys())
        print(f"Camera type: {camera_type}")

        if list(results['images'].keys())[0] == self.cam:
            filename = results['images'][self.cam]['img_path']
        else:
            return None
            #raise KeyError(f'Camera {self.cam} not found in {results["images"]}')
        

        try: 
            img_bytes = get(filename, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, flag=self.color_type, backend=self.imdecode_backend)
            #depth path : depthroot/(img_prefix - dataroot)+img_name.replace('.jpg', '.npy')
            depth_file_name = filename.split('/')[-1].replace('.jpg', '.npy')
            depth_path = os.path.join(self.depth_root + f'/samples/{self.cam}/' + depth_file_name)
            depth_meters = np.load(depth_path)
            #assert depth and img dimesions
            assert depth_meters.shape[0] == img.shape[0] and depth_meters.shape[1] == img.shape[1]
        except Exception as e:
            raise RuntimeError(f'Failed to load image {filename} from {self.backend_args}: {e}')
        
        results['img'] = img
        results['cam2img'] = results['images'][self.cam]['cam2img']
        results['depth_meters'] = depth_meters
        return results
        
        

@TRANSFORMS.register_module()
class AddFog(BaseTransform):
    def __init__(self, strategy: dict[str, float], total_epochs: None) -> None:
        self.strategy = strategy
        self.beta = strategy['beta'] if 'beta' in strategy else 0
        self.airlight = strategy['airlight'] if 'airlight' in strategy else None
        self.epoch_no = None
       
        if 'alpha' in self.strategy:
            self.alpha = self.strategy['alpha']
        else:
            self.alpha = 0.5
            print_log(f"Alpha is not provided, using default value: {self.alpha}", logger='current', level=logging.WARNING)
        
        # DESIGN CHOICE: Picking up total_epochs from strategy if not passed directly
        self.total_epochs = total_epochs if total_epochs is not None else self.strategy.get('total_epochs')

    def transform(self, results: dict) -> dict:
        img = results['img']
        depth_meters = results['depth_meters']
        if int(self.beta) == 0: # no fog, don't process at all
            return results
        if self.beta<=0.02:
            print_log(f"Beta is too small, unrealistic fog may be generated: {self.beta}", logger='current', level=logging.WARNING)
        

        #update beta based on training progress i.e epochs or iterations
        #get epoch_info and batch_info from results
        epoch_info = results['epoch_info']

        if self.strategy['type'] == 'linear':
            assert self.epoch_no is not None, "Epoch number is not set"
            # interpolate beta between betamin, betmax, using epoch or iterations by total iterations
            progress = self.epoch_no / self.total_epochs
            self.beta = self.strategy['betamin'] + (self.strategy['betamax'] - self.strategy['betamin']) * progress
            self.airlight = self.estimate_airlight(img)
            foggy_img = add_fog_beta(img, depth_meters, beta=self.beta, airlight=self.airlight)
            results['img'] = foggy_img
        elif self.strategy['type'] == 'probabilistic':
            # to make half of images foggy, toss coin with alpha note this work if only batch size >= 16 
            if np.random.rand() < self.alpha:
                self.beta = self.update_beta_curriculum()
                self.airlight = self.estimate_airlight(img)
                foggy_img = add_fog_beta(img, depth_meters, beta=self.beta, airlight=self.airlight)
                results['img'] = foggy_img
        elif self.strategy['type'] == 'loss_based':
            #use model_loss to update beta, gradient of loss w.r.t difficult d
            model_loss = results['model_loss']
            raise NotImplementedError
        else:
            print_log(f"Unknown strategy type: {self.strategy['type']}, using no fog ", logger='current', level=logging.WARNING)


        return results
    
    def estimate_airlight(self, img: np.ndarray) -> int:
        airlight, _ = estimate_airlight(img)
        return airlight
    


#Hooks won't update info in multi-worker mode, we need to make non-persistent worker for this to work
@HOOKS.register_module()
class InjectEpochHook(Hook):
    def before_train_epoch(self, runner):
        #couldn't see any metainfo in dataset class
        runner.train_dataloader.dataset.metainfo['epoch_info'] = runner.epoch
        dataset = runner.train_dataloader.dataset
        dataset.metainfo['epoch_info'] = runner.epoch # i don't think dataset has metainfo ?
        for transform in dataset.pipeline.transforms:
            if isinstance(transform, AddFog):
                transform.epoch_no = runner.epoch
                break
        if hasattr(runner.train_dataloader, 'persistent_workers'
                       ) and runner.train_dataloader.persistent_workers is True:
                runner.train_dataloader._DataLoader__initialized = False
                runner.train_dataloader._iterator = None
                self._restart_dataloader = True
    # def after_train_epoch(self, runner):
    #     runner.train_dataloader.dataset.metainfo['model_loss'] = runner.train_loss
