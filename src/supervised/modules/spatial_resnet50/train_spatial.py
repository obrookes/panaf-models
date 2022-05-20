import torch
import argparse
import configparser
import torchmetrics
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from panaf.datamodules import SupervisedPanAfDataModule
from src.supervised.models import ResNet50


class ActionClassifier(pl.LightningModule):
    def __init__(self, lr, weight_decay):
        super().__init__()

        self.save_hyperparameters()

        self.spatial_model = ResNet50()

        self.top1_train_accuracy = torchmetrics.Accuracy(top_k=1)
        self.top1_val_accuracy = torchmetrics.Accuracy(top_k=1)

        self.preds = []
        self.labels = []

    def training_step(self, batch, batch_idx):
        x, y = batch
        spatial_pred = self.spatial_model(x["spatial_sample"].permute(0, 2, 1, 3, 4))
        loss = F.cross_entropy(spatial_pred, y)
        top1_train_acc = self.top1_train_accuracy(spatial_pred, y)

        self.log(
            "top1_train_acc",
            top1_train_acc,
            logger=False,
            on_epoch=False,
            on_step=True,
            prog_bar=True,
        )
        return {"loss": loss}

    def training_epoch_end(self, outputs):

        # Log epoch acc
        top1_acc = self.top1_train_accuracy.compute()
        self.log(
            "train_top1_acc_epoch",
            top1_acc,
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
        spatial_pred = self.spatial_model(x["spatial_sample"].permute(0, 2, 1, 3, 4))
        loss = F.cross_entropy(spatial_pred, y)

        top1_val_acc = self.top1_val_accuracy(spatial_pred, y)

        self.log(
            "top1_val_acc",
            top1_val_acc,
            logger=False,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        return {"loss": loss}

    def validation_epoch_end(self, outputs):

        # Log epoch acc
        top1_acc = self.top1_val_accuracy.compute()
        self.log(
            "val_top1_acc_epoch",
            top1_acc,
            logger=True,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

        # Log epoch loss
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log(
            "val_loss_epoch",
            loss,
            logger=True,
            on_epoch=True,
            on_step=False,
            prog_bar=False,
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
    )
    wand_logger = WandbLogger(offline=True)

    if cfg.getboolean("remote", "slurm"):
        trainer = pl.Trainer(
            gpus=cfg.getint("trainer", "gpus"),
            num_nodes=cfg.getint("trainer", "num_nodes"),
            strategy=cfg.get("trainer", "strategy"),
            max_epochs=cfg.getint("trainer", "max_epochs"),
            stochastic_weight_avg=cfg.getboolean("trainer", "swa"),
            logger=wand_logger,
        )
    else:
        trainer = pl.Trainer(
            max_epochs=cfg.getint("trainer", "max_epochs"), fast_dev_run=5
        )
    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()
