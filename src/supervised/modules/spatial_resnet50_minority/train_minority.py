import torch
import argparse
import configparser
import torchmetrics
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from panaf.datamodules import SupervisedPanAfDataModule
from src.supervised.models import MinorityResNet50


class ActionClassifier(pl.LightningModule):
    def __init__(self, lr, weight_decay, freeze_backbone):
        super().__init__()

        self.save_hyperparameters()

        self.model = MinorityResNet50(freeze_backbone=freeze_backbone)

        self.ce_loss = nn.CrossEntropyLoss()

        # Training metrics
        self.top1_train_accuracy = torchmetrics.Accuracy(top_k=1)
        self.train_per_class_accuracy = torchmetrics.Accuracy(
            num_classes=6, average="macro"
        )
        # Validation metrics
        self.top1_val_accuracy = torchmetrics.Accuracy(top_k=1)
        self.val_per_class_accuracy = torchmetrics.Accuracy(
            num_classes=6, average="macro"
        )

    def forward(self, x):
        pred = self.model(x["spatial_sample"].permute(0, 2, 1, 3, 4))
        return pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)

        self.top1_train_accuracy(pred, y)
        self.train_per_class_accuracy(pred, y)

        loss = self.ce_loss(pred, y)

        return {"loss": loss}

    def training_epoch_end(self, outputs):

        # Log epoch acc
        self.log(
            "train_top1_acc_epoch",
            self.top1_train_accuracy,
            logger=True,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

        # Log epoch acc
        self.log(
            "train_per_class_acc_epoch",
            self.train_per_class_accuracy,
            logger=True,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log(
            "train_loss_epoch",
            loss,
            logger=True,
            on_epoch=True,
            on_step=False,
            prog_bar=False,
        )

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)

        loss = self.ce_loss(pred, y)

        self.top1_val_accuracy(pred, y)
        self.val_per_class_accuracy(pred, y)

        return {"loss": loss}

    def validation_epoch_end(self, outputs):

        # Log top-1 acc per epoch
        self.log(
            "val_top1_acc_epoch",
            self.top1_val_accuracy,
            logger=True,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

        # Log per class acc per epoch
        self.log(
            "val_per_class_acc",
            self.val_per_class_accuracy,
            logger=True,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config)

    data_module = SupervisedPanAfDataModule(cfg=cfg)

    model = ActionClassifier(
        lr=cfg.getfloat("hparams", "lr"),
        weight_decay=cfg.getfloat("hparams", "weight_decay"),
        freeze_backbone=cfg.getboolean("hparams", "freeze_backbone"),
    )
    wand_logger = WandbLogger(offline=True)

    val_top1_acc_checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/val_top1_acc", monitor="val_top1_acc_epoch", mode="max"
    )

    val_per_class_acc_checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/val_per_class_acc",
        monitor="val_per_class_acc_epoch",
        mode="max",
    )

    if cfg.get("remote", "slurm") == "ssd" or cfg.get("remote", "slurm") == "hdd":
        if not cfg.getboolean("mode", "test"):
            trainer = pl.Trainer(
                gpus=cfg.getint("trainer", "gpus"),
                num_nodes=cfg.getint("trainer", "num_nodes"),
                strategy=cfg.get("trainer", "strategy"),
                max_epochs=cfg.getint("trainer", "max_epochs"),
                stochastic_weight_avg=cfg.getboolean("trainer", "swa"),
                callbacks=[
                    val_top1_acc_checkpoint_callback,
                    val_per_class_acc_checkpoint_callback,
                ],
                logger=wand_logger,
            )
        else:
            trainer = pl.Trainer(
                gpus=cfg.getint("trainer", "gpus"),
                num_nodes=cfg.getint("trainer", "num_nodes"),
                strategy=cfg.get("trainer", "strategy"),
                max_epochs=cfg.getint("trainer", "max_epochs"),
                stochastic_weight_avg=cfg.getboolean("trainer", "swa"),
                logger=wand_logger,
                fast_dev_run=10,
            )
    else:
        trainer = pl.Trainer(
            gpus=cfg.getint("trainer", "gpus"),
            num_nodes=cfg.getint("trainer", "num_nodes"),
            strategy=cfg.get("trainer", "strategy"),
            max_epochs=cfg.getint("trainer", "max_epochs"),
            stochastic_weight_avg=cfg.getboolean("trainer", "swa"),
            fast_dev_run=10,
        )
    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()
