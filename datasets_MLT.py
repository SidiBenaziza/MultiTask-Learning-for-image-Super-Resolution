import numpy as np
from torch.utils.data import Dataset
import os 

def getLength(data_path):
    len=0
    for file in os.listdir(data_path):
        len+=1
    return len

def getDict(data_path):
    Dict={}
    idx=0

    for file in sorted(os.listdir(data_path)):
        Dict[idx]=os.path.join(data_path,file)
        idx+=1
    
    return Dict
def channelSplit(image):
    return np.dsplit(image,3)


class TrainDataset(Dataset):
    def __init__(self,data_path_hr,data_path_lr,data_path_hq,transform):
        self.root_dir_hr=data_path_hr
        self.root_dir_lr=data_path_lr
        self.root_dir_hq=data_path_hq
        self.dic_hr=getDict(data_path_hr)
        self.dic_lr=getDict(data_path_lr)
        self.dic_hq=getDict(data_path_hq)
        self.transform=transform


    def __getitem__(self, index):
        lr_img = np.load(self.dic_lr[index])/255.
        hr_img = np.load(self.dic_hr[index])/255.
        hq_img = np.load(self.dic_hq[index])/255.

        hr_img_y, _, _  = channelSplit(hr_img)
        lr_img_y, _, _  = channelSplit(lr_img)
        hq_img_y, _, _  = channelSplit(hq_img)
        if self.transform is 'transpose':
            
            hr_img_y = np.transpose(hr_img_y,(2,1,0))
            lr_img_y = np.transpose(lr_img_y,(2,1,0))
            hq_img_y = np.transpose(hq_img_y,(2,1,0))
            
        return  lr_img_y, hr_img_y, hq_img_y


    def __len__(self):
        return getLength(self.root_dir_hr)