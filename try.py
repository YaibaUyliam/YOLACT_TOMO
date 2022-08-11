import copy
import numpy as np 
import os 
# a = np.array([1, 4, 2, 6, 7])
# b = copy.deepcopy(a)
# id = 0 

# for value in b:
#     print(value)
#     if value > 4:
#         a = np.delete(a, id, 0)
#         id -= 1
#     id += 1        
# print(a)
import glob
a = glob.glob("/home/crist_tienngoc/TOMO/Multi_object/Dataset/Data_train/val/*.png")
for i in a:
    if os.path.exists(i + '.json'):
        print(i)