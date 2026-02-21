nuscenes_infos_train.pkl: training dataset, a dict contains two keys: metainfo and data_list. metainfo contains the basic information for the dataset itself, such as categories, dataset and info_version, while data_list is a list of dict, each dict (hereinafter referred to as info) contains all the detailed information of single sample as follows:

info[‘sample_idx’]: The index of this sample in the whole dataset.

info[‘token’]: Sample data token.

info[‘timestamp’]: Timestamp of the sample data.

info[‘ego2global’]: The transformation matrix from the ego vehicle to global coordinates. (4x4 list)

info[‘lidar_points’]: A dict containing all the information related to the lidar points.

info[‘lidar_points’][‘lidar_path’]: The filename of the lidar point cloud data.

info[‘lidar_points’][‘num_pts_feats’]: The feature dimension of point.

info[‘lidar_points’][‘lidar2ego’]: The transformation matrix from this lidar sensor to ego vehicle. (4x4 list)

info[‘lidar_sweeps’]: A list contains sweeps information (The intermediate lidar frames without annotations)

info[‘lidar_sweeps’][i][‘lidar_points’][‘data_path’]: The lidar data path of i-th sweep.

info[‘lidar_sweeps’][i][‘lidar_points’][‘lidar2ego’]: The transformation matrix from this lidar sensor to ego vehicle. (4x4 list)

info[‘lidar_sweeps’][i][‘lidar_points’][‘ego2global’]: The transformation matrix from the ego vehicle to global coordinates. (4x4 list)

info[‘lidar_sweeps’][i][‘lidar2sensor’]: The transformation matrix from the main lidar sensor to the current sensor (for collecting the sweep data). (4x4 list)

info[‘lidar_sweeps’][i][‘timestamp’]: Timestamp of the sweep data.

info[‘lidar_sweeps’][i][‘sample_data_token’]: The sweep sample data token.

info[‘images’]: A dict contains six keys corresponding to each camera: 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'. Each dict contains all data information related to corresponding camera.

info[‘images’][‘CAM_XXX’][‘img_path’]: The filename of the image.

info[‘images’][‘CAM_XXX’][‘cam2img’]: The transformation matrix recording the intrinsic parameters when projecting 3D points to each image plane. (3x3 list)

info[‘images’][‘CAM_XXX’][‘sample_data_token’]: Sample data token of image.

info[‘images’][‘CAM_XXX’][‘timestamp’]: Timestamp of the image.

info[‘images’][‘CAM_XXX’][‘cam2ego’]: The transformation matrix from this camera sensor to ego vehicle. (4x4 list)

info[‘images’][‘CAM_XXX’][‘lidar2cam’]: The transformation matrix from lidar sensor to this camera. (4x4 list)

info[‘instances’]: It is a list of dict. Each dict contains all annotation information of single instance. For the i-th instance:

info[‘instances’][i][‘bbox_3d’]: List of 7 numbers representing the 3D bounding box of the instance, in (x, y, z, l, w, h, yaw) order.

info[‘instances’][i][‘bbox_label_3d’]: A int indicate the label of instance and the -1 indicate ignore.

info[‘instances’][i][‘velocity’]: Velocities of 3D bounding boxes (no vertical measurements due to inaccuracy), a list has shape (2.).

info[‘instances’][i][‘num_lidar_pts’]: Number of lidar points included in each 3D bounding box.

info[‘instances’][i][‘num_radar_pts’]: Number of radar points included in each 3D bounding box.

info[‘instances’][i][‘bbox_3d_isvalid’]: Whether each bounding box is valid. In general, we only take the 3D boxes that include at least one lidar or radar point as valid boxes.

info[‘cam_instances’]: It is a dict containing keys 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'. For vision-based 3D object detection task, we split 3D annotations of the whole scenes according to the camera they belong to. For the i-th instance:

info[‘cam_instances’][‘CAM_XXX’][i][‘bbox_label’]: Label of instance.

info[‘cam_instances’][‘CAM_XXX’][i][‘bbox_label_3d’]: Label of instance.

info[‘cam_instances’][‘CAM_XXX’][i][‘bbox’]: 2D bounding box annotation (exterior rectangle of the projected 3D box), a list arrange as [x1, y1, x2, y2].

info[‘cam_instances’][‘CAM_XXX’][i][‘center_2d’]: Projected center location on the image, a list has shape (2,), .

info[‘cam_instances’][‘CAM_XXX’][i][‘depth’]: The depth of projected center.

info[‘cam_instances’][‘CAM_XXX’][i][‘velocity’]: Velocities of 3D bounding boxes (no vertical measurements due to inaccuracy), a list has shape (2,).

info[‘cam_instances’][‘CAM_XXX’][i][‘attr_label’]: The attr label of instance. We maintain a default attribute collection and mapping for attribute classification.

info[‘cam_instances’][‘CAM_XXX’][i][‘bbox_3d’]: List of 7 numbers representing the 3D bounding box of the instance, in (x, y, z, l, h, w, yaw) order.

info[‘pts_semantic_mask_path’]：The filename of the lidar point cloud semantic segmentation annotation.

Alredy made annotations from:
https://mmdetection3d.readthedocs.io/en/latest/user_guides/dataset_prepare.html

Nuscenes pkl file links:

wget -c https://download.openmmlab.com/mmdetection3d/data/nuscenes/nuscenes_infos_train.pkl
wget -c https://download.openmmlab.com/mmdetection3d/data/nuscenes/nuscenes_infos_val.pkl
