from importlib.machinery import PathFinder
import os 
import glob 

path_file = '/home/crist_tienngoc/TOMO/Multi_object/yolact_edge/multi_object_2/weight'
list_file = os.listdir(path_file)

for name_file in list_file:
    if '52' not in name_file:
        os.remove(os.path.join(path_file, name_file))