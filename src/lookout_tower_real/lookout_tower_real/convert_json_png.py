import os

# path
directory = "/home/william/Datasets/floorspace"

def labelme_json_to_dataset(json_path):
    os.system("labelme_json_to_dataset "+json_path+" -o "+json_path.replace(".","_"))

for filename in os.listdir(directory):
    if filename.endswith(".json"):
        labelme_json_to_dataset(os.path.join(directory, filename))
    else:
        continue