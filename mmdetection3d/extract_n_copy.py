import mmengine
import shutil
import os

# Paths
src_root = 'data/nuscenes/samples/CAM_FRONT'
dst_root = 'data/nuscenes_test_subset'
old_ann = 'data/nuscenes/nuscenes_infos_val.pkl'
new_ann = os.path.join(dst_root, 'mini_infos_val.pkl')

os.makedirs(dst_root, exist_ok=True)

# Load original
infos = mmengine.load(old_ann)
subset_size = 5
mini_infos = {'metainfo': infos['metainfo'], 'data_list': infos['data_list'][:subset_size]}

# Copy files and fix paths
for item in mini_infos['data_list']:
    # We only care about CAM_FRONT for this demo
    rel_path = item['images']['CAM_FRONT']['img_path']
    src_path = os.path.join(src_root, rel_path)
    dst_path = os.path.join(dst_root, rel_path)
    
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy(src_path, dst_path)
    print(f"Copied: {rel_path}")

# Save the mini pickle
mmengine.dump(mini_infos, new_ann)
print(f"\nCreated mini-annotation file at: {new_ann}")
