import torch
import argparse
import configparser
import torchmetrics
import pytorch_lightning as pl
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from panaf.datamodules import SupervisedPanAfDataModule
from src.supervised.callbacks.custom_metrics import PerClassAccuracy
from configparser import NoOptionError
from train_byol import ActionClassifier


class BYOLFinetuner(pl.LightningModule):
    def __init__(self, lr, weight_decay, ckpt_path, freeze_backbone, out_features):
        super().__init__()

        self.save_hyperparameters()

        # TODO: automatically pass model name
        self.backbone = ActionClassifier.load_from_checkpoint(ckpt_path)
        self.fc = nn.Linear(in_features=2048, out_features=out_features)

        if self.hparams.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.model = nn.Sequential(self.backbone, self.fc)

        # Loss
        self.ce_loss = nn.CrossEntropyLoss()

        # Training metrics
        self.train_top1_acc = torchmetrics.Accuracy(top_k=1)
        self.train_avg_per_class_acc = torchmetrics.Accuracy(
            num_classes=out_features, average="macro"
        )
        self.train_per_class_acc = torchmetrics.Accuracy(
            num_classes=out_features, average="none"
        )

        # Validation metrics
        self.val_top1_acc = torchmetrics.Accuracy(top_k=1)
        self.val_avg_per_class_acc = torchmetrics.Accuracy(
            num_classes=out_features, average="macro"
        )
        self.val_per_class_acc = torchmetrics.Accuracy(
            num_classes=out_features, average="none"
        )

    def forward(self, x):
        pred = self.model(x)
        return pred

    def training_step(self, batch, batch_idx):

        x, y = batch
        pred = self(x["spatial_sample"].permute(0, 2, 1, 3, 4))

        self.train_top1_acc(pred, y)
        self.train_avg_per_class_acc(pred, y)
        self.train_per_class_acc.update(pred, y)

        loss = self.ce_loss(pred, y)

        return {"loss": loss}

    def training_epoch_end(self, outputs):

        self.log(
            "train_top1_acc",
            self.train_top1_acc,
            logger=True,
            prog_bar=True,
        )

        self.log(
            "train_avg_per_class_acc",
            self.train_avg_per_class_acc,
            logger=True,
            prog_bar=True,
        )

        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log(
            "train_loss",
            loss,
            logger=True,
            prog_bar=False,
        )

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x["spatial_sample"].permute(0, 2, 1, 3, 4))

        self.val_top1_acc(pred, y)
        self.val_avg_per_class_acc(pred, y)
        self.val_per_class_acc.update(pred, y)

        loss = self.ce_loss(pred, y)

        return {"loss": loss}

    def validation_epoch_end(self, outputs):

        self.log(
            "val_top1_acc",
            self.val_top1_acc,
            logger=True,
            prog_bar=True,
        )

        # Log per class acc per epoch
        self.log(
            "val_avg_per_class_acc",
            self.val_avg_per_class_acc,
            logger=True,
            prog_bar=True,
            rank_zero_only=True,
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

    wandb_logger = WandbLogger(offline=True)
    data_module = SupervisedPanAfDataModule(cfg=cfg)

    which_classes = cfg.get("dataset", "classes") if not NoOptionError else "all"
    no_of_classes = 9 if which_classes == "all" else 6

    model = BYOLFinetuner(
        lr=cfg.getfloat("hparams", "lr"),
        weight_decay=cfg.getfloat("hparams", "weight_decay"),
        ckpt_path=cfg.get("trainer", "ckpt"),
        freeze_backbone=cfg.getboolean("hparams", "freeze_backbone"),
        out_features=no_of_classes,
    )

    per_class_acc_callback = PerClassAccuracy(which_classes=which_classes)

    val_top1_acc_checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/val_top1_acc", monitor="val_top1_acc", mode="max"
    )

    val_per_class_acc_checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/val_per_class_acc",
        monitor="val_avg_per_class_acc",
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
                    per_class_acc_callback,
                ],
                logger=wandb_logger,
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
    else:
        trainer = pl.Trainer(
            gpus=cfg.getint("trainer", "gpus"),
            num_nodes=cfg.getint("trainer", "num_nodes"),
            strategy=cfg.get("trainer", "strategy"),
            max_epochs=cfg.getint("trainer", "max_epochs"),
            stochastic_weight_avg=cfg.getboolean("trainer", "swa"),
            callbacks=[per_class_acc_callback],
            fast_dev_run=5,
        )
    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()
