import torch.nn as nn
from models.MM_fusion_layer import LinearProbe, TwoTower, PromptPAR, CrossFormer
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
            model = PromptPAR(clip_model=clip_model)
            
            # model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    elif cfg['MODEL']['NAME'] == 'linaer_probe':
         clip_model, ViT_preprocess = load(name=cfg['CLIP']['PRETRAINED_ViT_NAME'], 
                                              device=f"cuda:{cfg['TRAINER']['DEVICES']}",
                                              download_root='data')
         
         model = LinearProbe(clip_model=clip_model)
         
    elif cfg['MODEL']['NAME'] == 'TWO_TOWER':
         clip_model, ViT_preprocess = load(name=cfg['CLIP']['PRETRAINED_ViT_NAME'], 
                                              device=f"cuda:{cfg['TRAINER']['DEVICES']}",
                                              download_root='data')
         
         model = TwoTower(clip_model=clip_model)

    elif cfg['MODEL']['NAME'] == 'CROSS_FORMER':
         clip_model, ViT_preprocess = load(name=cfg['CLIP']['PRETRAINED_ViT_NAME'], 
                                              device=f"cuda:{cfg['TRAINER']['DEVICES']}",
                                              download_root='data')
         
         model = CrossFormer(clip_model=clip_model)

    return model, clip_model


def make_loss(sample_weight):
    criterion = CEL_Sigmoid(sample_weight, attr_idx=cfg['MODEL']['ATTRIBUTE_NUM'])
    return criterion


