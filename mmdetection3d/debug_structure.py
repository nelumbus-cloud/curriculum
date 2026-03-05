import pickle
import sys

# Load the file
path = 'data/nuscenes/nuscenes_infos_train.pkl'
print(f"Loading {path}...")
with open(path, 'rb') as f:
    data = pickle.load(f)

# Extract the list
if 'data_list' in data:
    data_list = data['data_list']
    print("Found 'data_list' key.")
elif 'infos' in data:
    data_list = data['infos']
    print("Found 'infos' key.")
else:
    data_list = data
    print("Structure is a direct list.")

# INSPECT THE FIRST ITEM
if len(data_list) == 0:
    print("ERROR: List is empty!")
    sys.exit(1)

item = data_list[0]
print("\n--- DIAGNOSTIC REPORT ---")
print(f"Type of first item: {type(item)}")

if isinstance(item, dict):
    print(f"Keys available: {list(item.keys())}")
    
    # Check 'images' specifically
    if 'images' in item:
        print("\n'images' dictionary found. Keys inside 'images':")
        print(list(item['images'].keys()))
        
        if 'CAM_FRONT' in item['images']:
            print("\nContent of 'CAM_FRONT':")
            print(item['images']['CAM_FRONT'])
        else:
            print("\nCRITICAL: 'CAM_FRONT' NOT found in 'images'.")
    else:
        print("\nCRITICAL: 'images' key NOT found in item.")
        
    # Check for flat file path
    if 'img_path' in item:
        print(f"\nFound flat 'img_path': {item['img_path']}")
    elif 'filename' in item:
        print(f"\nFound flat 'filename': {item['filename']}")

else:
    print(f"Content: {item}")
    print("CRITICAL: Item is not a dictionary! It might be a Token string.")

print("\n--- END REPORT ---")
