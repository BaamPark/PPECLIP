import torch
import yaml
import argparse
from utils.misc import seed_everything
from dataset.make_dataloader import make_dataloader
from models.build import build_model, make_loss
from models.promptpar import TransformerClassifier
from models.model import LightningModel
from utils.config import Config, get_config
from utils.model_utils import freeze_model, make_optimizer, make_scheduler
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

seed_everything(seed=42)

def main():
    config_path = "config/base.yaml"
    log_name = "test"
    Config.set_config_file(config_path)
    cfg = get_config()

    train_loader, val_loader, test_loader = make_dataloader()
    model = LightningModel()

    logger = TensorBoardLogger("lightning_logs", name=log_name)
    trainer = Trainer(
        accelerator=cfg["TRAINER"]["ACCELERATOR"],
        devices=cfg["TRAINER"]["DEVICES"],
        max_epochs=cfg["HYPERPARAM"]["NUM_EPOCH"],
        logger=logger
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )


if __name__ == "__main__":
    main()