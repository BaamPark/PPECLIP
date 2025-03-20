import torch
import os
from PIL import Image
import json
from torch.utils.data.dataset import Dataset
import torchvision.transforms as T
import numpy as np


attr_words = [
    'A person wearing a gown completely',
    'A person not wearing a gown',
    'A person wearing a gown incorrectly',
    'A person wearing a helmet',
    'A person wearing goggles',
    'A person wearing a face shield',
    'A person not wearing eyewear',
    'A person wearing an N95 mask',
    'A person wearing a regular mask',
    'A person not wearing a mask',
    'A person with two visible hands wearing two gloves',
    'A person with one visible hand wearing one glove',
    'A person with two visible hands wearing no gloves',
    'A person with one visible glove, the other hand not visible',
    'A person with one visible hand, the other hand not visible'
]


class PPEmultilabelDataset(Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path, 'r') as file:
            json_data = json.load(file)

        self.json_data = json_data
        self.transform = transform
        self.attributes = attr_words
        self.attr_num = len(attr_words)


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
        # ppe_map = {'gc':0, 'ga':1, 'gi':2, 'pr':3, 'gg':4, 'fc':5, 'fi':6, 'sg':7, 'ea':8, 'nc':9, 'rc':10, 'ma':11, 'hchc':12, 'hcha':13, 'haha':14, 'hc':15, 'ha':16}
        ppe_map = {'gc':0, 'ga':1, 'gi':2, 'pr':3, 'gg':4, 'fc':5, 'ea':6, 'nc':7, 'rc':8, 'ma':9, 'hchc':10, 'hcha':11, 'haha':12, 'hc':13, 'ha':14}
        # ppe_map = {'gc':0, 'ga':1, 'gi':2, 'pr':3, 'gg':4, 'fc':5, 'sg': 6, 'ea':7, 'nc':8, 'rc':9, 'ma':10, 'hchc':11, 'hcha':12, 'haha':13, 'hc':14, 'ha':15}
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