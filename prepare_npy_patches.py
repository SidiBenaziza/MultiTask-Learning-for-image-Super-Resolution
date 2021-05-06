import argparse
import cv2
import numpy as np
from utils import convert_rgb_to_y
import numpy as np
import os
import re
from tqdm import tqdm



def getYUVFrame(f, width, height):
    ysize = width*height
    csize = (width)//2 *(height)//2
    f_size = width*height*3//2
    yuv = np.frombuffer((f.read(f_size)), dtype = np.uint8)
    
    Y = np.reshape(yuv[0:ysize          ], (height, width))
    U = np.reshape(yuv[ysize+0*csize:ysize+1*csize], (height//2, width//2)).repeat(2, 0).repeat(2, 1)
    V = np.reshape(yuv[ysize+1*csize:ysize+2*csize], (height//2, width//2)).repeat(2, 0).repeat(2, 1)
    YUV = np.dstack([Y, U, V])

    return YUV


def resize_with_padding(img, patch_size):
    
    expected_height, expected_width = img.shape[0:2]    

    while expected_width % patch_size is not 0 :
        expected_width+=1
    
    while expected_height % patch_size is not 0 :
        expected_height+=1
        
    delta_height = expected_height - img.shape[0]
    delta_width = expected_width - img.shape[1]
    pad_height = delta_height // 2
    pad_width = delta_width // 2
    
    img_padded = cv2.copyMakeBorder(img,pad_height,pad_height,pad_width,pad_width,cv2.BORDER_REFLECT)
    
    return img_padded 

def get_patches(img_padded, patch_size):
    
    img_data_array=[]

    for i in range(img_padded.shape[0] // patch_size):

        for j in range(img_padded.shape[1] // patch_size):


                patch_center = np.array([ (patch_size// 2) + patch_size*i ,(patch_size // 2) + patch_size*j])
                patch_x = int(patch_center[0] - patch_size / 2.)
                patch_y = int(patch_center[1] - patch_size / 2.)

                patch_image = img_padded[patch_x:patch_x+patch_size, patch_y:patch_y+patch_size]
                img_data_array.append(patch_image)


    patches = np.array(img_data_array,np.uint8)
    
    return patches

def train_set(args):

    path = args.images_dir
    patch_size = args.patch_size
    list_path = os.listdir(path)
    path_bar = tqdm(list_path)

    os.makedirs(path+'_npy')

    for f in path_bar: 
        if 'yuv' in f : 

            filename = path + '/' + f
            img_YUV = open(filename, 'rb')

            dimensions = re.search(r'^.*_(.*)\.yuv$', filename).group(1).split('x')
            np_yuv_im = getYUVFrame(img_YUV, int(dimensions[0]), int(dimensions[1]))

            img_padded = resize_with_padding(np_yuv_im,patch_size)
           
            patches = get_patches(img_padded,patch_size)

            i=0
            for patch in patches : 
                name_np = f.split('.yuv')[0] +str(i) +'.npy'
                np.save(path+'_npy/'+name_np, patch)
                i+=1
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--patch_size', type=int, default=256)
  
    args = parser.parse_args()

    train_set(args)

    
   