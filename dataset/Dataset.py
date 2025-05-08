import torch
import os
from PIL import Image
import json
from torch.utils.data.dataset import Dataset
import torchvision.transforms as T
import numpy as np
from utils.config import get_config
from scipy.io import loadmat

cfg = get_config()

class PPEmultilabelDataset(Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path, 'r') as file:
            json_data = json.load(file)

        self.json_data = json_data
        self.transform = transform
        self.attributes = cfg['ATTRIBUTES']
        self.attr_num = len(cfg['ATTRIBUTES'])
        self.label = np.vstack([self._encode_label(entry) for entry in self.json_data])

    def __getitem__(self, index):
        imagepath = self.json_data[index]['path']
        img = Image.open(imagepath)
        img = img.convert('RGB')
        # head_img = img.crop(self.json_data[index]['headbox'])
        cropped_img = img.crop(self.json_data[index]['bbox'])
        if self.transform is not None:
            cropped_img = self.transform(cropped_img)
            
        return cropped_img, self.label[index]
    
    def __len__(self):
        return len(self.json_data)

    def _encode_label(self, json_query):
        ppe_map = cfg['DATASET']['PPE_MAPS']
        onehot_enocding = torch.zeros(len(ppe_map))
        if json_query['gown'] != 'na':
            gown_code = ppe_map[json_query['gown']]
            onehot_enocding[gown_code] = 1

        if json_query['eyewear'] != 'na':
            eyewear_code = ppe_map[json_query['eyewear']]
            onehot_enocding[eyewear_code] = 1

        if json_query['mask'] != 'na':
            mask_code = ppe_map[json_query['mask']]
            onehot_enocding[mask_code] = 1

        if json_query['glove'] != 'na':
            glove_code = ppe_map[json_query['glove']]
            onehot_enocding[glove_code] = 1

        return onehot_enocding
    
    
class PA100k(Dataset):
    def __init__(self, split, transform=None, target_transform=None):
        pa100k_data = loadmat(os.path.join('data/PA100k/annotation.mat'))
        train_image_name = [pa100k_data['train_images_name'][i][0][0] for i in range(80000)]
        val_image_name = [pa100k_data['val_images_name'][i][0][0] for i in range(10000)]
        test_image_name = [pa100k_data['test_images_name'][i][0][0] for i in range(10000)]
        img_id = train_image_name + val_image_name + test_image_name
        attr_label = np.concatenate((pa100k_data['train_label'], pa100k_data['val_label'], pa100k_data['test_label']), axis=0)

        self.transform = transform
        self.target_transform = target_transform

        self.root_path = 'data/PA100k/data'
        
        if split == 'train':
            self.img_idx = np.arange(0, 80000)
        elif split == 'val':
            self.img_idx = np.arange(80000, 90000)
        elif split == 'test':
            self.img_idx = np.arange(90000, 100000)

        if isinstance(self.img_idx, list):
            self.img_idx = self.img_idx[0]
        self.img_num = self.img_idx.shape[0]
        self.img_id = [img_id[i] for i in self.img_idx]

        self.label = attr_label[self.img_idx]
        self.label_all = self.label

    def __getitem__(self, index):
        imgname, gt_label = self.img_id[index], self.label[index]
        imgpath = os.path.join(self.root_path, imgname)
        img_pil = Image.open(imgpath)
        
        if self.transform is not None:
            img_pil = self.transform(img_pil)

        gt_label = gt_label.astype(np.float32)
        
        if self.target_transform is not None:
            gt_label = self.transform(gt_label)
        
        return img_pil, gt_label

    def __len__(self):
        return len(self.img_id)