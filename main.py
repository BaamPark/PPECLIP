import argparse
from utils.misc import seed_everything
parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true")
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--logdir", type=str, required=True)
args = parser.parse_args()
seed_everything(seed=42)

from utils.config import Config, get_config
Config.set_config_file(args.config)

from dataset.make_dataloader import make_dataloader
from models.model import LightningModel
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import torch


def main():
    cfg = get_config()

    train_loader, val_loader, test_loader = make_dataloader()
    model = LightningModel()

    logger = TensorBoardLogger("lightning_logs", name=args.logdir)
    trainer = Trainer(
        accelerator=cfg["TRAINER"]["ACCELERATOR"],
        devices=cfg["TRAINER"]["DEVICES"],
        max_epochs=cfg["HYPERPARAM"]["NUM_EPOCH"],
        logger=logger
    )

    if not args.test:
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )

    else:
        trainer.test(
                model=model,
                ckpt_path = cfg["TRAINER"]["CHECKPOINT"],
                dataloaders=test_loader
        )


if __name__ == "__main__":
    main()