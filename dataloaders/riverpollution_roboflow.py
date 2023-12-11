import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pdb import set_trace as stop
from PIL import Image
import torchvision.transforms.functional as TF
import pandas as pd
from dataloaders.data_utils import get_unk_mask_indices

category_info = {'Aeration':0, 'Discolouration_Colour':1, 'Discolouration_Outfall':2,'Fish': 3,'Modified_Channel':4, 'Obsruction':5,
                 'Outfall':6, 'Outfall_Aeration':7, 'Outfall_Screen':8, 'Outfall_Spilling':9,
                 'Rubbish':10, 'Sensor':11, 'Wildlife_Algal':12, 'Wildlife_Birds':13,
                 'Wildlife_Others':14}

class rproboflowDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir='./data/We-do-care-the-rivers-18', flag = 'train' , image_transform=None,known_labels=0,testing=False):
        self.flag = flag
        if flag == 'train':
            self.base_dir = os.path.join(img_dir,'train')
            anno_path = os.path.join(self.base_dir,'_classes.csv')
        elif flag == 'valid':
            self.base_dir = os.path.join(img_dir,'valid')
            anno_path = os.path.join(self.base_dir,'_classes.csv')
        elif flag == 'test':
            self.base_dir = os.path.join(img_dir,'test')
            anno_path = os.path.join(self.base_dir,'_classes.csv')
        elif flag == 'user_defined':
            self.base_dir = os.path.join(img_dir)
            anno_path = os.path.join(self.base_dir,'_classes.csv')

        self.img_path  = []
        annotation = pd.read_csv(anno_path)
        for i in range(len(annotation)):
            self.img_path.append(os.path.join(self.base_dir,annotation.iloc[i]['filename']))
        self.num_labels = len(category_info)
        self.known_labels = known_labels
        self.testing = testing
        self.labels = []
        for i in range(len(annotation)):
            label_vector = np.zeros(self.num_labels)
            for j in range(1,self.num_labels+1):
                if annotation.iloc[i,j] == 1:
                    label_vector[j-1] = 1.0
            self.labels.append(label_vector)
        self.labels = np.array(self.labels).astype(int)
        self.image_transform = image_transform
        self.epoch = 1
    def __getitem__(self, index):
        name = self.img_path[index]
        image = Image.open(name).convert('RGB')
        if self.image_transform:
            image = self.image_transform(image)
        labels = torch.Tensor(self.labels[index])
        unk_mask_indices = get_unk_mask_indices(image,self.testing,self.num_labels,self.known_labels,self.epoch)
        mask = labels.clone()
        mask.scatter_(0,torch.Tensor(unk_mask_indices).long() , -1)
        sample = {}
        sample['image'] = image
        sample['labels'] = labels
        sample['mask'] = mask
        sample['imageIDs'] = str(name)
        return sample
    def __len__(self):
        return len(self.img_path)


