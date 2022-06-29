import torch
import argparse
import configparser
import pytorch_lightning as pl
from torch import nn
from pytorch_lightning.loggers import WandbLogger
from panaf.datamodules import SupervisedPanAfDataModule
from src.supervised.models import ResNet50, TemporalResNet50


class ActionClassifier(pl.LightningModule):
    def __init__(self, lr, weight_decay, freeze_backbone):
        super().__init__()

        self.save_hyperparameters()

        self.spatial_model = ResNet50(freeze_backbone=freeze_backbone)
        self.dense_model = ResNet50(freeze_backbone=freeze_backbone)
        self.temporal_model = TemporalResNet50(freeze_backbone=freeze_backbone)

        # Loss
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        spatial_pred = self.spatial_model(x["spatial_sample"].permute(0, 2, 1, 3, 4))
        dense_pred = self.dense_model(x["dense_sample"].permute(0, 2, 1, 3, 4))
        temporal_pred = self.temporal_model(x["flow_sample"].permute(0, 2, 1, 3, 4))
        pred = (spatial_pred + dense_pred + temporal_pred) / 3
        return pred

    def training_step(self, batch, batch_idx):

        x, y = batch
        pred = self(x)
        loss = self.ce_loss(pred, y)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):

        x, y = batch
        pred = self(x)
        loss = self.ce_loss(pred, y)

        return {"loss": loss}

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

    if cfg.get("remote", "slurm") == "ssd" or cfg.get("remote", "slurm") == "hdd":
        if not cfg.getboolean("mode", "test"):
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
            fast_dev_run=5,
        )
    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()
