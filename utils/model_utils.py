import torch
from utils.config import get_config
import torch.optim as optim

cfg = get_config()

def make_optimizer(params, lr, weight_decay):
    optimizer = optim.SGD(params, lr, momentum=0.9, weight_decay=weight_decay)
    return optimizer

def make_scheduler(optimizer, lr):
    num_warmup_epochs = cfg['HYPERPARAM']['WARMUP_EPOCH']
    total_epochs = cfg['HYPERPARAM']['NUM_EPOCH']
    
    def warmup_lambda(epoch):
        if epoch < num_warmup_epochs:
            return 0.01 + 0.99 * (epoch / num_warmup_epochs)  # Linearly increase LR
        else:
            return 1  # Continue with the cosine scheduler
    
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=total_epochs - num_warmup_epochs,
        eta_min=lr * 0.01
    )

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[num_warmup_epochs]
    )

def freeze_model(model, clip_model):
    mm_params=[]
    #Freeze parameters other than category headers and prompts
    for name, param in model.named_parameters():
        if any(keyword in name for keyword in cfg['MODEL']['PARAMETERS_TO_BE_UPDATED']):
            mm_params+= [{
            "params": [param],
            "lr": cfg['HYPERPARAM']['LR'],
            "weight_decay": cfg['HYPERPARAM']['WEIGHT_DECAY']
            }]
        else:
            param.requires_grad = False
    clip_params=[]
    for name, param in clip_model.named_parameters():
        if any(keyword in name for keyword in cfg['CLIP']['PARAMETERS_TO_BE_UPDATED']):
            print(name, param.requires_grad)
            clip_params+= [{
            "params": [param],
            "lr": cfg['HYPERPARAM']['CLIP_LR'],
            "weight_decay": cfg['HYPERPARAM']['CLIP_WEIGHT_DECAY']
            }]
        else:
            param.requires_grad = False

    return mm_params, clip_params

def ratio2weight(targets, ratio):
    ratio = torch.from_numpy(ratio).type_as(targets)
    pos_weights = targets * (1 - ratio)
    neg_weights = (1 - targets) * ratio
    weights = torch.exp(neg_weights + pos_weights)

    # for RAP dataloader, targets element may be 2, with or without smooth, some element must great than 1
    weights[targets > 1] = 0.0

    return weights

def count_parameters(model,model2,selected_param_names,selected_param_names2):
    total_params = sum(p.numel() for p in model.parameters())
    total_params += sum(p.numel() for p in model2.parameters())
    selected_params1 = []
    selected_params2 = []
    for name, param in model.named_parameters():
        if any(param_name in name for param_name in selected_param_names):
            selected_params1.append(param)
    for name, param in model2.named_parameters():
        if any(param_name in name for param_name in selected_param_names2):
            selected_params2.append(param)  
      
    selected_params_count1 = sum(p.numel() for p in selected_params1)
    selected_params_count2 = sum(p.numel() for p in selected_params2)
    trainable_percentage = ((selected_params_count1+selected_params_count2) / total_params) * 100 if total_params > 0 else 0
    print(f"MM-former trainable params: {selected_params_count1} || prompt trainable params: {selected_params_count2}")
    print(f"trainable params: {(selected_params_count1+selected_params_count2)} || all params: {total_params} || trainable%: {trainable_percentage:.12f}")