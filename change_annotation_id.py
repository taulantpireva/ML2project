import os

# Paths
BASE_PATH = 'D:/ZHAW/ML2/ML2project/food-101'
META_PATH = os.path.join(BASE_PATH, 'meta/asdf.txt')
CATEGORIES_PATH = os.path.join(BASE_PATH, 'category.txt')
DATASET_PATHS = [os.path.join(BASE_PATH, folder) for folder in ['train',  'valid']]
#'test',

# Read category mapping
category_mapping = {}
with open(CATEGORIES_PATH, 'r') as f:
    for line in f.readlines()[1:]:
        class_id, class_name = line.strip().split(maxsplit=1)
        category_mapping[class_name] = int(class_id) - 1  # 0-indexed

# Read meta information
meta_info = {}
with open(META_PATH, 'r') as f:
    for line in f.readlines():
        class_name, img_id = line.strip().split('/')
        meta_info[img_id] = class_name

# Update annotation files
for dataset_path in DATASET_PATHS:
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.txt'):
                img_id = file.split('.')[0]
                if img_id in meta_info:
                    class_name = meta_info[img_id]
                    class_id = category_mapping.get(class_name)
                    annotation_file = os.path.join(root, file)
                    with open(annotation_file, 'r') as f:
                        lines = f.readlines()
                    with open(annotation_file, 'w') as f:
                        for line in lines:
                            parts = line.strip().split()
                            parts[0] = str(class_id)
                            f.write(' '.join(parts) + '\n')

print("Annotations updated with correct class IDs.")
