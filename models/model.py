from pytorch_lightning import LightningModule
from torch import nn
import torch
from torchmetrics.classification import MultilabelF1Score, MultilabelAUROC, MultilabelRecall, MultilabelSpecificity, MultilabelPrecision
from utils.config import get_config
from utils.model_utils import make_optimizer, make_scheduler, freeze_model
from models.build import build_model, make_loss

cfg = get_config()

class LightningModel(LightningModule):
    def __init__(self):
        super(LightningModel, self).__init__()
        model, clip = build_model() #model represents VLM
        model_params, clip_params = freeze_model(model, clip)
        self.model = model
        self.clip = clip
        self.model_params = model_params
        self.clip_params = clip_params
        self.criterion = make_loss()
        self.automatic_optimization = False
        
    def forward(self, x):
        logit, final_similarity = self.model(x, clip_model=self.clip)
        return logit

    def training_step(self, batch, batch_idx):
        clip_optimizer, model_optimizer = self.optimizers()
        img, lbl = batch
        logits, similarity = self.model(img, self.clip)

        if cfg['CLIP']['USE_GLOBAL_LOCAL_SIMILARITY']:
            classifier_loss = self.criterion(logits, lbl)
            clip_loss = self.criterion(similarity, lbl)
            loss = classifier_loss + 0.5 * clip_loss
        else:
            loss = self.criterion(logits, lbl)
        
        model_optimizer.zero_grad()
        clip_optimizer.zero_grad()

        self.manual_backward(loss)
        # Clip gradients to avoid exploding gradient problem
        self.clip_gradients(model_optimizer, gradient_clip_val=10.0, gradient_clip_algorithm="norm")
        self.clip_gradients(clip_optimizer, gradient_clip_val=10.0, gradient_clip_algorithm="norm")

        model_optimizer.step()
        clip_optimizer.step()

        self.log("train_step_loss", loss, on_step=True, on_epoch=False)
        return loss


    def validation_step(self, batch, batch_idx):
        img, lbl = batch
        logits, similarity = self.model(img, self.clip)

        if cfg['CLIP']['USE_GLOBAL_LOCAL_SIMILARITY']:
            classifier_loss = self.criterion(logits, lbl)
            clip_loss = self.criterion(similarity, lbl)
            loss = classifier_loss + 0.5 * clip_loss
        else:
            loss = self.criterion(logits, lbl)

        self.log('val_step_loss', loss, on_step=True, on_epoch=False)
        self.log(f"val_epoch_loss", loss, on_step=False, on_epoch=True, prog_bar=True)


    def configure_optimizers(self): #https://pytorch-lightning.readthedocs.io/en/1.5.10/common/optimizers.html#use-multiple-optimizers-like-gans:~:text=and%20step.-,Use%20multiple%20optimizers%20(like%20GANs),-To%20use%20multiple
        clip_optimizer = make_optimizer(self.clip_params, cfg['HYPERPARAM']['CLIP_LR'], cfg['HYPERPARAM']['CLIP_WEIGHT_DECAY'])
        clip_scheduler = make_scheduler(clip_optimizer, cfg['HYPERPARAM']['CLIP_LR'], )
        model_optimizer = make_optimizer(self.model_params, cfg['HYPERPARAM']['LR'], cfg['HYPERPARAM']['WEIGHT_DECAY'])
        model_scheduler = make_scheduler(model_optimizer, cfg['HYPERPARAM']['LR'])
        return [clip_optimizer, model_optimizer], [clip_scheduler, model_scheduler]