import torch
import argparse
import configparser
import torchmetrics
import pytorch_lightning as pl
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from panaf.datamodules import SupervisedPanAfDataModule
from src.supervised.models import ResNet50, TemporalResNet50
from src.supervised.callbacks.custom_metrics import PerClassAccuracy


class ActionClassifier(pl.LightningModule):
    def __init__(self, lr, weight_decay, freeze_backbone):
        super().__init__()

        self.save_hyperparameters()

        self.spatial_model = ResNet50(freeze_backbone=freeze_backbone)
        self.temporal_model = TemporalResNet50(freeze_backbone=freeze_backbone)

        # Loss
        self.ce_loss = nn.CrossEntropyLoss()

        # Training metrics
        self.train_top1_acc = torchmetrics.Accuracy(top_k=1)
        self.train_avg_per_class_acc = torchmetrics.Accuracy(
            num_classes=9, average="macro"
        )
        self.train_per_class_acc = torchmetrics.Accuracy(num_classes=9, average="none")

        # Validation metrics
        self.val_top1_acc = torchmetrics.Accuracy(top_k=1)
        self.val_avg_per_class_acc = torchmetrics.Accuracy(
            num_classes=9, average="macro"
        )
        self.val_per_class_acc = torchmetrics.Accuracy(num_classes=9, average="none")

    def forward(self, x):
        spatial_pred = self.spatial_model(x["spatial_sample"].permute(0, 2, 1, 3, 4))
        temporal_pred = self.temporal_model(x["flow_sample"].permute(0, 2, 1, 3, 4))
        pred = (spatial_pred + temporal_pred) / 2
        return pred

    def training_step(self, batch, batch_idx):

        x, y = batch
        pred = self(x)

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
            rank_zero_only=True,
        )

        self.log(
            "train_per_class_acc",
            self.train_avg_per_class_acc,
            logger=True,
            prog_bar=True,
            rank_zero_only=True,
        )

        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log(
            "train_loss",
            loss,
            logger=True,
            prog_bar=False,
            rank_zero_only=True,
        )

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)

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
            rank_zero_only=True,
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

    data_module = SupervisedPanAfDataModule(cfg=cfg)

    wandb_logger = WandbLogger(offline=True)

    model = ActionClassifier(
        lr=cfg.getfloat("hparams", "lr"),
        weight_decay=cfg.getfloat("hparams", "weight_decay"),
        freeze_backbone=cfg.getboolean("hparams", "freeze_backbone"),
    )

    avg_per_class_acc_callback = PerClassAccuracy(
        which_classes=cfg.get("dataset", "classes")
    )

    val_top1_acc_checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/val_top1_acc", monitor="val_top1_acc", mode="max"
    )

    val_per_class_acc_checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/val_avg_per_class_acc",
        monitor="val_per_class_acc",
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
            callbacks=[avg_per_class_acc_callback],
            logger=wandb_logger,
            fast_dev_run=5,
        )
    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()
