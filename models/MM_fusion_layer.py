import torch.nn as nn
import torch
from models.clip_utils import tokenize
from models.vit import *
from utils.config import get_config
import numpy as np

cfg = get_config()
cfg_model = cfg['MODEL']
cfg_clip = cfg['CLIP']


class PromptPAR(nn.Module):
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


class LinearProbe(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.attr_num = cfg_model['ATTRIBUTE_NUM']
        self.word_embed = nn.Linear(clip_model.visual.output_dim, cfg_model['TEXT_DIMENSION'])
        self.norm = nn.LayerNorm(cfg_model['TEXT_DIMENSION'])
        self.weight_layer = nn.ModuleList([nn.Linear(cfg_model['TEXT_DIMENSION'], 1) for i in range(self.attr_num)])
        self.dim = cfg_model['TEXT_DIMENSION']
        self.text = tokenize(cfg['ATTRIBUTES']).to(f"cuda:{cfg['TRAINER']['DEVICES']}")
        self.bn = nn.BatchNorm1d(self.attr_num)
        self.use_region_split = cfg_clip['USE_REGION_SPLIT'] #bool
        self.use_mmformer = cfg_model['USE_MULTIMODAL_TRANSFORMER']
        self.fusion_len = 1 + 256 + self.attr_num + cfg_clip['REGION_SPLIT_NUM'] + cfg_clip['VISUAL_PROMPT_NUM']
        self.linear_layer = nn.Linear(self.fusion_len, self.attr_num)


    def forward(self,imgs,clip_model):
        b_s=imgs.shape[0]
        clip_image_features,all_class,attenmap=clip_model.visual(imgs.type(clip_model.dtype))
        text_features = clip_model.encode_text(self.text).to(f"cuda:{cfg['TRAINER']['DEVICES']}").float()
        textual_features = self.word_embed(text_features).expand(b_s, self.attr_num, self.dim)
        x = torch.cat([textual_features,clip_image_features], dim=1)
        x = x.permute(0, 2, 1)
        x= self.linear_layer(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        logits = torch.cat([self.weight_layer[i](x[:, i, :]) for i in range(self.attr_num)], dim=1)
        bn_logits = self.bn(logits)
        return bn_logits, None
    

class TwoTower(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.attr_num = cfg_model["ATTRIBUTE_NUM"]
        self.word_embed = nn.Linear(clip_model.visual.output_dim, cfg_model['TEXT_DIMENSION'])
        self.clip_model = clip_model  # keep reference for inference‑time use
        self.text = tokenize(cfg["ATTRIBUTES"]).to(f"cuda:{cfg['TRAINER']['DEVICES']}")
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        

    def forward(self, imgs, clip_model):
        clip_model = self.clip_model
        visual_feature, all_class,attenmap=clip_model.visual(imgs.type(clip_model.dtype))
        visual_feature = visual_feature.float()
        text_features = clip_model.encode_text(self.text).to(f"cuda:{cfg['TRAINER']['DEVICES']}").float()
        textual_features = self.word_embed(text_features)
        
        #if you just want simple dot product instead of cosine similarity, uncommnet below and comment the rest
        # logits = visual_feature @ textual_features.T
        
        logit_scale = self.logit_scale.exp()
        visual_feature = visual_feature / visual_feature.norm(dim=-1, keepdim=True)
        textual_features = textual_features / textual_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * visual_feature @ textual_features.T  

        return logits, None


class CrossAttentionFusion(nn.Module):
    """
    Pure multi‑head cross‑attention.
    • Q  : attribute prototypes  [B, C, D]
    • K,V: visual tokens        [B, T, D]
    returns cross‑attended attrs [B, C, D]
    """
    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn  = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, txt_attr: torch.Tensor, vis_tokens: torch.Tensor):
        txt_attr   = txt_attr.float()
        vis_tokens = vis_tokens.float()
        fused, _ = self.attn(txt_attr, vis_tokens, vis_tokens)  # Q,K,V
        fused = fused + txt_attr                                # residual
        return self.ffn(fused)                                  # LN‑FFN
# ─────────────────────────────────────────────────────────────
class CrossFormer(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model
        self.attr_num = cfg_model['ATTRIBUTE_NUM']
        self.dim      = cfg_model['TEXT_DIMENSION']

        self.word_embed = nn.Linear(clip_model.visual.output_dim, self.dim)
        self.text_toks  = tokenize(cfg['ATTRIBUTES']).to(f"cuda:{cfg['TRAINER']['DEVICES']}")

        self.fusion  = CrossAttentionFusion(self.dim, heads=cfg_model['NUM_CROSS_ATTENTION_HEAD'])
        self.heads = nn.ModuleList(
            [nn.Linear(self.dim, 1) for _ in range(self.attr_num)]
        )
        self.bn = nn.BatchNorm1d(self.attr_num)

    # ---------------------------------------------------------
    def forward(self, imgs, clip_model=None):
        clip_model = self.clip if clip_model is None else clip_model
        B = imgs.size(0)

        vis_tokens, _, _ = clip_model.visual(imgs.type(clip_model.dtype))
        txt_proto = self.word_embed(
            clip_model.encode_text(self.text_toks).float()
        ).unsqueeze(0).expand(B, -1, -1)          # [B,C,D]

        fused = self.fusion(txt_proto, vis_tokens) # [B,C,D]
        logits = torch.cat([self.heads[i](fused[:, i, :]) for i in range(self.attr_num)], dim=1)
        return self.bn(logits), None