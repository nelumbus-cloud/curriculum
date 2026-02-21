config_dir = 'configs/fcos3d'
_base_ = f'{config_dir}/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d.py'

# train pipeline
train_pipeline = [
    dict(type="AddFog")
]
