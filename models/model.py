from pytorch_lightning import LightningModule
from torch import nn
import torch
from torchmetrics.classification import MultilabelF1Score, MultilabelAUROC, MultilabelRecall, MultilabelSpecificity, MultilabelPrecision
from utils.config import get_config
from utils.model_utils import make_optimizer, make_scheduler, freeze_model
from models.build import build_model, make_loss
from torchmetrics.classification import MultilabelF1Score, MultilabelAUROC, MultilabelRecall, MultilabelSpecificity, MultilabelPrecision
import pandas as pd
import numpy as np

cfg = get_config()

class LightningModel(LightningModule):
    def __init__(self, sample_weight):
        super(LightningModel, self).__init__()
        model, clip = build_model() #model represents VLM
        model_params, clip_params = freeze_model(model, clip)
        self.model = model
        self.clip = clip
        self.model_params = model_params
        self.clip_params = clip_params
        self.criterion = make_loss(sample_weight)
        self.automatic_optimization = False
        self.validation_step_outputs = {"target": [], 
                                        "output": []}
        
    def forward(self, x):
        logit, final_similarity = self.model(x, clip_model=self.clip)
        return logit


    def training_step(self, batch, batch_idx):
        clip_optimizer, model_optimizer = self.optimizers()
        clip_scheduler, model_scheduler = self.lr_schedulers()

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
        self.log("lr_model", model_scheduler._get_lr(self.current_epoch)[0], on_step=True)
        self.log("lr_clip", clip_scheduler._get_lr(self.current_epoch)[0], on_step=True)
        # self.scheduler.get_lr()[0], on_step=True
        return loss


    def validation_step(self, batch, batch_idx):
        img, lbl = batch
        logits, similarity = self.model(img, self.clip)

        if cfg['CLIP']['USE_REGION_SPLIT']:
            classifier_loss = self.criterion(logits, lbl)
            clip_loss = self.criterion(similarity, lbl)
            loss = classifier_loss + 0.5 * clip_loss
        else:
            loss = self.criterion(logits, lbl)

        probs = torch.sigmoid(logits)

        if batch_idx == 0:
            self.validation_step_outputs["output"] = probs
            self.validation_step_outputs["target"] = lbl
        else:
            self.validation_step_outputs["output"] = torch.cat((self.validation_step_outputs["output"], probs), dim=0)
            self.validation_step_outputs["target"] = torch.cat((self.validation_step_outputs["target"], lbl), dim=0)
        
        self.log('val_step_loss', loss, on_step=True, on_epoch=False)
        self.log(f"val_epoch_loss", loss, on_step=False, on_epoch=True, prog_bar=True)


    def test_step(self, batch, batch_idx):
        img, lbl = batch
        logits, similarity = self.model(img, self.clip)

        if cfg['CLIP']['USE_REGION_SPLIT']:
            classifier_loss = self.criterion(logits, lbl)
            clip_loss = self.criterion(similarity, lbl)
            loss = classifier_loss + 0.5 * clip_loss
        else:
            loss = self.criterion(logits, lbl)

        probs = torch.sigmoid(logits)

        if batch_idx == 0:
            self.validation_step_outputs["output"] = probs
            self.validation_step_outputs["target"] = lbl
        else:
            self.validation_step_outputs["output"] = torch.cat((self.validation_step_outputs["output"], probs), dim=0)
            self.validation_step_outputs["target"] = torch.cat((self.validation_step_outputs["target"], lbl), dim=0)
        
        # self.log('test_step_loss', loss, on_step=True, on_epoch=False)
        self.log(f"test_epoch_loss", loss, on_step=False, on_epoch=True, prog_bar=True)


    def on_validation_epoch_end(self):
        y_pred = self.validation_step_outputs["output"] #[[0.3, 0.1, 0.6], [0.2, 0.7, 0.1]]
        y_true = self.validation_step_outputs["target"] #[2, 1]
        f1 = MultilabelF1Score(num_labels=cfg["MODEL"]["ATTRIBUTE_NUM"], threshold=0.5, average=None).to(self.device) #https://lightning.ai/docs/torchmetrics/stable/classification/f1_score.html#:~:text=target)%0Atensor(%5B0.6667%2C%200.6667%2C%201.0000%5D)-,Example%20(preds%20is%20float%20tensor)%3A,-%3E%3E%3E
        f1_weighted = MultilabelF1Score(num_labels=cfg["MODEL"]["ATTRIBUTE_NUM"], threshold=0.5, average="weighted").to(self.device)
        auroc = MultilabelAUROC(num_labels=cfg["MODEL"]["ATTRIBUTE_NUM"], average=None).to(self.device)
        recall = MultilabelRecall(num_labels=cfg["MODEL"]["ATTRIBUTE_NUM"], average=None).to(self.device)
        specificity = MultilabelSpecificity(num_labels=cfg["MODEL"]["ATTRIBUTE_NUM"], average=None).to(self.device)
        precision = MultilabelPrecision(num_labels=cfg["MODEL"]["ATTRIBUTE_NUM"], average=None).to(self.device)
        # Compute metrics
        y_true = y_true.to(torch.long)
        F1 = f1(y_pred, y_true)
        F1_WEIGHTED = f1_weighted(y_pred, y_true)
        AUROC = auroc(y_pred, y_true)
        RECALL = recall(y_pred, y_true)
        SPECIFICITY = specificity(y_pred, y_true)
        PRECISION = precision(y_pred, y_true)

        # for i, (each_f1, each_auroc, each_recall, each_specificity, each_precision) in enumerate(zip(F1, AUROC, RECALL, SPECIFICITY, PRECISION)):
        #     self.log(f"val_F1_cls{i}", each_f1)
        #     # self.log(f"val_auc_cls{i}", each_auroc)
        #     self.log(f"val_rec_cls{i}", each_recall)
        #     # self.log(f"val_spe_cls{i}", each_specificity)
        #     self.log(f"val_pre_cls{i}", each_precision)

        self.log("val_mean_F1", F1.mean()) #macro average
        self.log("val_mean_F1_weighted", F1_WEIGHTED)
        self.log("val_mean_auc", AUROC.mean())
        self.log("val_mean_rec", RECALL.mean())
        self.log("val_mean_spe", SPECIFICITY.mean())
        self.log("val_mean_pre", PRECISION.mean())

        # flush validation_step_output
        self.validation_step_outputs = {"target": [], 
                                        "output": []}
        
    def on_test_epoch_end(self):
        y_pred = self.validation_step_outputs["output"] #[[0.3, 0.1, 0.6], [0.2, 0.7, 0.1]]
        y_true = self.validation_step_outputs["target"] #[2, 1]
        f1 = MultilabelF1Score(num_labels=cfg["MODEL"]["ATTRIBUTE_NUM"], threshold=0.5, average=None).to(self.device) #https://lightning.ai/docs/torchmetrics/stable/classification/f1_score.html#:~:text=target)%0Atensor(%5B0.6667%2C%200.6667%2C%201.0000%5D)-,Example%20(preds%20is%20float%20tensor)%3A,-%3E%3E%3E
        f1_weighted = MultilabelF1Score(num_labels=cfg["MODEL"]["ATTRIBUTE_NUM"], threshold=0.5, average="weighted").to(self.device)
        auroc = MultilabelAUROC(num_labels=cfg["MODEL"]["ATTRIBUTE_NUM"], average=None).to(self.device)
        recall = MultilabelRecall(num_labels=cfg["MODEL"]["ATTRIBUTE_NUM"], average=None).to(self.device)
        specificity = MultilabelSpecificity(num_labels=cfg["MODEL"]["ATTRIBUTE_NUM"], average=None).to(self.device)
        precision = MultilabelPrecision(num_labels=cfg["MODEL"]["ATTRIBUTE_NUM"], average=None).to(self.device)
        # Compute metrics
        y_true = y_true.to(torch.long)
        F1 = f1(y_pred, y_true)
        F1_WEIGHTED = f1_weighted(y_pred, y_true)
        AUROC = auroc(y_pred, y_true)
        RECALL = recall(y_pred, y_true)
        SPECIFICITY = specificity(y_pred, y_true)
        PRECISION = precision(y_pred, y_true)

        # for i, (each_f1, each_auroc, each_recall, each_specificity, each_precision) in enumerate(zip(F1, AUROC, RECALL, SPECIFICITY, PRECISION)):
        #     self.log(f"val_F1_cls{i}", each_f1)
        #     # self.log(f"val_auc_cls{i}", each_auroc)
        #     self.log(f"val_rec_cls{i}", each_recall)
        #     # self.log(f"val_spe_cls{i}", each_specificity)
        #     self.log(f"val_pre_cls{i}", each_precision)

        self.log("val_mean_F1", F1.mean())
        self.log("val_mean_F1_weighted", F1_WEIGHTED)
        self.log("val_mean_auc", AUROC.mean())
        self.log("val_mean_rec", RECALL.mean())
        self.log("val_mean_spe", SPECIFICITY.mean())
        self.log("val_mean_pre", PRECISION.mean())

        # y_pred = y_pred.cpu().numpy()
        # y_true = y_true.cpu().numpy()

        # pred_gt = np.concatenate([y_pred, y_true], axis=1)
        # columns = list(cfg["DATASET"]["PPE_MAPS"].keys())
        # pred_columns, gt_columns = [f'pred_{col}' for col in columns], [f'gt_{col}' for col in columns]
        # df = pd.DataFrame(pred_gt, columns=pred_columns+gt_columns)
        # df.to_excel('results/pred_gt/predictions.xlsx', index=False)
    
    def configure_optimizers(self): #https://pytorch-lightning.readthedocs.io/en/1.5.10/common/optimizers.html#use-multiple-optimizers-like-gans:~:text=and%20step.-,Use%20multiple%20optimizers%20(like%20GANs),-To%20use%20multiple
        clip_optimizer = make_optimizer(self.clip_params, cfg['HYPERPARAM']['CLIP_LR'], cfg['HYPERPARAM']['CLIP_WEIGHT_DECAY'])
        clip_scheduler = make_scheduler(clip_optimizer, cfg['HYPERPARAM']['CLIP_LR'], warmup_t=cfg['HYPERPARAM']['WARMUP_EPOCH'])
        model_optimizer = make_optimizer(self.model_params, cfg['HYPERPARAM']['LR'], cfg['HYPERPARAM']['WEIGHT_DECAY'])
        model_scheduler = make_scheduler(model_optimizer, cfg['HYPERPARAM']['LR'], warmup_t=cfg['HYPERPARAM']['WARMUP_EPOCH'])
        return [clip_optimizer, model_optimizer], [{"scheduler": clip_scheduler, "interval": "epoch"}, {"scheduler": model_scheduler, "interval": "epoch"}]
    
    #need to override when using timm scheduler or custom scheulder
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)  # timm's scheduler need the epoch valu