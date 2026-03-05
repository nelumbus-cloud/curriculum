import mmengine

info_list = mmengine.load('data/nuscenes/nuscenes_infos_train.pkl')
print(f"Total samples in inf : {len(info_list['data_list'])}")

for i in range(3):
#    img_path = info_list['data_list'][i]['images']['CAM_FRONT']
    info = info_list['data_list'][i]
    print(info['lidar_points'])
    print("*" * 20)
