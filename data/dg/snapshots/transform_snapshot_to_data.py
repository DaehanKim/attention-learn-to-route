'''Change dataset format for attention model'''

import numpy as np
import os 
import pickle


files = [item for item in os.listdir('.') if item.endswith('.npy')]
arr_dict = {}
for f in files:
    arr = np.load(f)
    new_fname = "../{}.pkl".format(f.split('.')[0])
    with open(new_fname,'wb') as f:
        pickle.dump([arr.tolist()], f)


