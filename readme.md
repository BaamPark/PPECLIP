### 1. Prepare prerequisite

``` bash
data
├── jx_vit_base_p16_224-80ecf9dd.pth
├── label_mat
│   ├── train_labels.pkl
│   └── val_labels.pkl
├── labels
│   ├── attribute_label_split_train_0303_sgIsEa.json
│   └── attribute_label_test_0303_sgIsEa.json
├── ViT-B-16.pt
└── ViT-L-14.pt
```
- Pre-trained CLIP, `ViT-L-14.pt` and `ViT-L-14.pt` will be downloaded automatically.
- Download Pre-trained ViT through https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth
- download json
- run `python tools/make_labelmat.py` to get pkl


This work is Unified Prompt Tuning