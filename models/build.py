import torch.nn as nn
import torch
import torch.nn as nn
from models.promptpar import TransformerClassifier
from models.MM_fusion_layer import LinearProbe
from models.clip_utils import load
from models.clip import build_clip
from utils.config import get_config
from loss.CE_loss import *
import pickle


cfg = get_config()

def build_model():
    if cfg['MODEL']['NAME'] == 'base':
            clip_model, ViT_preprocess = load(name=cfg['CLIP']['PRETRAINED_ViT_NAME'], 
                                              device=f"cuda:{cfg['TRAINER']['DEVICES']}",
                                              download_root='data')
            
            # clip_model = build_clip(checkpoint['clip_model'])
            model = TransformerClassifier(clip_model=clip_model)
            
            # model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    elif cfg['MODEL']['NAME'] == 'linaer_probe':
         clip_model, ViT_preprocess = load(name=cfg['CLIP']['PRETRAINED_ViT_NAME'], 
                                              device=f"cuda:{cfg['TRAINER']['DEVICES']}",
                                              download_root='data')
         
         model = LinearProbe(clip_model=clip_model)
    return model, clip_model


def make_loss():
    with open(cfg['DATASET']['TRAIN_LABEL_MAT'], "rb") as f:
        labels = pickle.load(f)

    sample_weight = labels.numpy().mean(0)
    criterion = CEL_Sigmoid(sample_weight, attr_idx=cfg['MODEL']['ATTRIBUTE_NUM'])
    return criterion


