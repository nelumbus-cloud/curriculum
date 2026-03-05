from mmdet3d.registry import TRANSFORMS, HOOKS
from mmcv.transforms import BaseTransform
from mmcv.transforms import LoadImageFromFile
from mmengine.logging import MessageHub


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
        self.ignore_empty = True

    def transform(self, results: dict) -> dict:
        #results["img_info"] contain info of sample idx.
        camera_type = None
        if self.cam in results['images']:
            camera_type = self.cam
            filename = results['images'][camera_type]['img_path']
            results['cam2img'] = results['images'][camera_type]['cam2img']
        elif len(results['images']) == 1:
            camera_type = list(results['images'].keys())[0]
            filename = results['images'][camera_type]['img_path']
            results['cam2img'] = results['images'][camera_type]['cam2img']
        else:
            raise KeyError(f'Camera {self.cam} not found in {results["images"].keys()}')

        try: 
            img_bytes = get(filename, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, flag=self.color_type, backend=self.imdecode_backend)
            #depth path : depthroot/(img_prefix - dataroot)+img_name.replace('.jpg', '.npy')
            depth_file_name = filename.split('/')[-1].replace('.jpg', '.npy')
            depth_path = os.path.join(self.depth_root + f'/samples/{camera_type}/' + depth_file_name)
            depth_meters = np.load(depth_path)
            #assert depth and img dimesions
            assert depth_meters.shape[0] == img.shape[0] and depth_meters.shape[1] == img.shape[1]
        except Exception as e:
            print(f"FAILED: filename={filename}, error={e}") 
            if self.ignore_empty:
                return None
            else:
                raise RuntimeError(f'Failed to load image {filename} from {self.backend_args}: {e}')
        
        results['img'] = img
        results['depth_meters'] = depth_meters
        results['img_shape'] = img.shape[:2]  
        results['ori_shape'] = img.shape[:2] 
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
            pass
            #return results
        if self.beta<=0.02:
            print_log(f"Beta is too small, unrealistic fog may be generated: {self.beta}", logger='current', level=logging.WARNING)
        

        #update beta based on training progress i.e epochs or iterations
        #get epoch_info and batch_info from results
        message_hub = MessageHub.get_current_instance()
        epoch_no = message_hub.get_info('epoch')

        if self.strategy['type'] == 'linear':
            assert epoch_no is not None, "Epoch number is not set"
            # interpolate beta between betamin, betmax, using epoch or iterations by total iterations
            progress = epoch_no / self.total_epochs
            self.beta = self.strategy['betamin'] + (self.strategy['betamax'] - self.strategy['betamin']) * progress
            if self.epoch_no and epoch_no > self.epoch_no:  # only print at the beginning of training
                print_log(f"Adding fog with beta level {self.beta}", logger='current', level=logging.INFO)
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

        self.epoch_no = epoch_no
        return results
    
    def estimate_airlight(self, img: np.ndarray) -> int:
        airlight, _ = estimate_airlight(img)
        return airlight
    

