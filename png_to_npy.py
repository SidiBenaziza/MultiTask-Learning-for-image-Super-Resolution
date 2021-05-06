import os
import numpy as np
import imageio
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-f", help="folder")
args = parser.parse_args()

path = args.f
list_path = os.listdir(path)
path_bar = tqdm(list_path)

os.makedirs(path+'_npy')


for f in path_bar:
    if 'png' in f:
        img=imageio.imread(path+'/'+f)
        name_np = f.split('.png')[0]+'.npy'
        np.save(path+'_npy/'+name_np, img)
