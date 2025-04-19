import torch.nn as nn
import torch
from models.clip_utils import tokenize
from models.vit import *
from utils.config import get_config

cfg = get_config()
cfg_model = cfg['MODEL']
cfg_clip = cfg['CLIP']

class TransformerClassifier(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.attr_num = cfg_model['ATTRIBUTE_NUM']
        self.word_embed = nn.Linear(clip_model.visual.output_dim, cfg_model['TEXT_DIMENSION'])
        vit = vit_base()
        vit.load_param(cfg_model['PRETRAINED_DIR'])
        self.norm = vit.norm
        self.weight_layer = nn.ModuleList([nn.Linear(cfg_model['TEXT_DIMENSION'], 1) for i in range(self.attr_num)])
        self.dim = cfg_model['TEXT_DIMENSION']
        self.text = tokenize(cfg['ATTRIBUTES']).to(f"cuda:{cfg['TRAINER']['DEVICES']}")
        self.bn = nn.BatchNorm1d(self.attr_num)
        self.use_region_split = cfg_clip['USE_REGION_SPLIT'] #bool
        self.use_mmformer = cfg_model['USE_MULTIMODAL_TRANSFORMER']
        self.fusion_len = 1 + 256 + self.attr_num + cfg_clip['REGION_SPLIT_NUM'] + cfg_clip['VISUAL_PROMPT_NUM']

        if not self.use_mmformer:
            print('Without MM-former, Using MLP Instead')
            self.linear_layer = nn.Linear(self.fusion_len, self.attr_num)
        else:
            self.blocks = vit.blocks[-cfg_model['MM_LAYERS_NUM']:]

    def forward(self,imgs,clip_model):
        b_s=imgs.shape[0]
        clip_image_features,all_class,attenmap=clip_model.visual(imgs.type(clip_model.dtype))
        text_features = clip_model.encode_text(self.text).to(f"cuda:{cfg['TRAINER']['DEVICES']}").float()
        if self.use_region_split:
            final_similarity,logits_per_image = clip_model.forward_aggregate(all_class,text_features)
        else : 
            final_similarity = None
        textual_features = self.word_embed(text_features).expand(b_s, self.attr_num, self.dim)
        x = torch.cat([textual_features,clip_image_features], dim=1)
        
        if self.use_mmformer:
            for blk in self.blocks:
                x = blk(x)
        else :# using linear layer fusion
            x = x.permute(0, 2, 1)
            x= self.linear_layer(x)
            x = x.permute(0, 2, 1)
            
        x = self.norm(x)
        logits = torch.cat([self.weight_layer[i](x[:, i, :]) for i in range(self.attr_num)], dim=1)
        bn_logits = self.bn(logits)
        
        return bn_logits,final_similarity