import torch
import os
from PIL import Image
import json
from torch.utils.data.dataset import Dataset
import torchvision.transforms as T
import numpy as np
from utils.config import get_config

cfg = get_config()

class PPEmultilabelDataset(Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path, 'r') as file:
            json_data = json.load(file)

        self.json_data = json_data
        self.transform = transform
        self.attributes = cfg['ATTRIBUTES']
        self.attr_num = len(cfg['ATTRIBUTES'])

    def __getitem__(self, index):
        imagepath = self.json_data[index]['path']
        img = Image.open(imagepath)
        img = img.convert('RGB')
        # head_img = img.crop(self.json_data[index]['headbox'])
        cropped_img = img.crop(self.json_data[index]['bbox'])
        if self.transform is not None:
            cropped_img = self.transform(cropped_img)

        attribute_label = self._encode_label(self.json_data[index])
        return cropped_img, attribute_label
    
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
    
    
def get_transform():
    height = 224
    width = 224
    normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    train_transform = T.Compose([
        T.Resize((height, width)),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])

    valid_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize
    ])

    return train_transform, valid_transform