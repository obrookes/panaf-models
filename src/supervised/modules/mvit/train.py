import torch
import argparse
import configparser
import torchmetrics
import pytorch_lightning as pl
from torch import nn
from einops import rearrange
from kornia.geometry.transform import resize
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from panaf.datamodules import SupervisedPanAfDataModule
from src.supervised.models import MViT
from src.supervised.callbacks.custom_metrics import PerClassAccuracy
from configparser import NoOptionError


class ActionClassifier(pl.LightningModule):
    def __init__(self, lr, weight_decay, model_name):
        super().__init__()

        self.save_hyperparameters()

        print(f"==> Initialising {model_name}!")
        self.mvit = MViT(model_name=model_name)

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

    def on_after_batch_transfer(self, batch, dataloader_idx):
        x, y = batch
        x = rearrange(x["spatial_sample"], "b t c w h -> b c t w h")
        x = resize(input=x, size=(224, 224))
        return x, y

    def training_step(self, batch, batch_idx):

        x, y = batch

        pred = self.mvit(x.contiguous())
        loss = self.ce_loss(pred, y)

        self.train_top1_acc(pred, y)
        self.train_avg_per_class_acc(pred, y)
        self.train_per_class_acc.update(pred, y)

        return {"loss": loss}

    def training_epoch_end(self, outputs):

        self.log(
            "train_top1_acc",
            self.train_top1_acc,
            logger=True,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

        self.log(
            "train_avg_per_class_acc",
            self.train_avg_per_class_acc,
            logger=True,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log(
            "train_loss",
            loss,
            logger=True,
            on_epoch=True,
            on_step=False,
            prog_bar=False,
        )

    def validation_step(self, batch, batch_idx):

        x, y = batch
        pred = self.mvit(x.contiguous())
        loss = self.ce_loss(pred, y)

        self.val_top1_acc(pred, y)
        self.val_avg_per_class_acc(pred, y)
        self.val_per_class_acc.update(pred, y)

        return {"loss": loss}

    def validation_epoch_end(self, outputs):

        # Log top-1 acc per epoch
        self.log(
            "val_top1_acc",
            self.val_top1_acc,
            logger=True,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

        # Log per class acc per epoch
        self.log(
            "val_avg_per_class_acc",
            self.val_avg_per_class_acc,
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
        model_name=cfg.get("model", "name"),
    )

    wand_logger = WandbLogger(offline=True)

    which_classes = cfg.get("dataset", "classes") if not NoOptionError else "all"
    per_class_acc_callback = PerClassAccuracy(which_classes=which_classes)

    val_top1_acc_checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{cfg.get('model', 'name')}/val_top1_acc",
        monitor="val_top1_acc",
        mode="max",
    )

    val_per_class_acc_checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{cfg.get('model', 'name')}/val_per_class_acc",
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
            # strategy=cfg.get("trainer", "strategy"),
            max_epochs=cfg.getint("trainer", "max_epochs"),
            stochastic_weight_avg=cfg.getboolean("trainer", "swa"),
            callbacks=[
                val_top1_acc_checkpoint_callback,
                val_per_class_acc_checkpoint_callback,
                per_class_acc_callback,
            ],
            fast_dev_run=2,
        )
    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()
