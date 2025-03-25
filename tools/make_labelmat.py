import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset.PPEDataset import PPEmultilabelDataset, get_transform, PPEmultilabelDataset12
from tqdm import tqdm
import pickle
import torch

train_tsfm, valid_tsfm = get_transform() 
train_set = PPEmultilabelDataset12(json_path="data/labels/attribute_label_split_train_0325.json", transform=train_tsfm)
valid_set = PPEmultilabelDataset12(json_path="data/labels/attribute_label_test_0325.json", transform=valid_tsfm) 


train_labels = []
for img, lbl in tqdm(train_set, desc="iterating trainset"):
    train_labels.append(lbl)

train_labels_tensor = torch.stack(train_labels)

val_labels = []
for img, lbl in tqdm(valid_set, desc="iterating trainset"):
    val_labels.append(lbl)

val_labels_tensor = torch.stack(val_labels)

with open("data/label_mat/train_labels_12cls.pkl", "wb") as f:
    pickle.dump(train_labels_tensor, f)

with open("data/label_mat/val_labels_12cls.pkl", "wb") as f:
    pickle.dump(val_labels_tensor, f)
