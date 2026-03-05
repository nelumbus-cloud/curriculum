_base_ = [
    '../mmdetection3d/configs/_base_/datasets/nus-mono3d.py', 
    '../mmdetection3d/configs/_base_/models/fcos3d.py',
    '../mmdetection3d/configs/_base_/schedules/mmdet-schedule-1x.py', 
    '../mmdetection3d/configs/_base_/default_runtime.py'
]

class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

#switching version to mini, change the following
data_root = 'data/nuscenes_mini/'
#data_root = 'data/nuscenes/'
depth_root = 'data/nuscenes_depth_meters'
version = 'v1.0-mini'
#usually change batch size
batch_size = 4



custom_imports = dict(imports=['modules.load_data'], allow_failed_imports=False)

beta_min, beta_max, total_epochs, num_workers = 0.01, 0.05, 12, 2


backend_args = None

model = dict(
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0],
        bgr_to_rgb=False, pad_size_divisor=32),
    backbone=dict(
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)))

train_pipeline = [
    dict(type='LoadSingleImageWithDepth', backend_args=backend_args, depth_root=depth_root),
    dict(type='LoadAnnotations3D', with_bbox=True, with_label=True, with_attr_label=True,
         with_bbox_3d=True, with_label_3d=True, with_bbox_depth=True),
    dict(type='AddFog', strategy=dict(type='linear', betamin=beta_min, betamax=beta_max), total_epochs=total_epochs),
    dict(type='mmdet.Resize', scale=(1600, 900), keep_ratio=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='Pack3DDetInputs', keys=['img', 'gt_bboxes', 'gt_bboxes_labels', 'attr_labels',
                                       'gt_bboxes_3d', 'gt_labels_3d', 'centers_2d', 'depths']),
]
test_pipeline = [
    dict(type='LoadSingleImageWithDepth', depth_root=depth_root),
    dict(type='mmdet.Resize', scale_factor=1.0),
    dict(type='Pack3DDetInputs', keys=['img']),
]

#dataset quirks
train_dataloader = dict(
    batch_size=batch_size, num_workers=num_workers, persistent_workers=False,
    dataset=dict(data_root=data_root, ann_file='nuscenes_infos_train.pkl',
                 pipeline=train_pipeline, metainfo=dict(version=version, classes=class_names)))
test_dataloader = dict(
    dataset=dict(data_root=data_root, ann_file='nuscenes_infos_val.pkl',
                 pipeline=test_pipeline, metainfo=dict(version=version, classes=class_names)))
val_dataloader = dict(
    dataset=dict(data_root=data_root, ann_file='nuscenes_infos_val.pkl',
                 pipeline=test_pipeline, metainfo=dict(version=version, classes=class_names)))



val_evaluator = dict(data_root=data_root, jsonfile_prefix='./work_dirs/fcos3d/results', ann_file=data_root + 'nuscenes_infos_val.pkl')
test_evaluator = val_evaluator

optim_wrapper = dict(
    optimizer=dict(lr=0.002),
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.),
    clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(type='LinearLR', start_factor=1/3, by_epoch=True, begin=0, end=1), 
    dict(type='MultiStepLR', begin=0, end=12, by_epoch=True, milestones=[8, 11], gamma=0.1)
]