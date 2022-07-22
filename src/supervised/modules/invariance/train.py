import torch
import argparse
import configparser
import torchmetrics
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from typing import Callable
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from panaf.datamodules import SupervisedPanAfPairsDataModule
from src.self_supervised.augmentations.simclr_augs import (
    SimCLRTrainDataTransform,
    SimCLREvalDataTransform,
)
from configparser import NoOptionError
from src.self_supervised.callbacks.custom_metrics import PerClassAccuracy
from src.self_supervised.models.resnets import ResNet50
from src.supervised.callbacks.online_evaluator import SSLOnlineEvaluator
from pl_bolts.optimizers import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay


class ActionClassifier(pl.LightningModule):
    def __init__(
        self,
        gpus: int,
        num_samples: int,
        batch_size: int,
        num_nodes: int = 1,
        in_features: int = 2048,
        hidden_features: int = 4096,
        out_features: int = 128,
        warmup_steps: int = 10,
        max_epochs: int = 100,
        temperature: float = 0.1,
        exclude_bn_bias: bool = False,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.left_model = ResNet50(freeze_backbone=True, out_features=2048)
        self.right_model = ResNet50(freeze_backbone=True, out_features=2048)

        self.projector = nn.Sequential(
            nn.Linear(
                in_features=in_features, out_features=hidden_features, bias=False
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=hidden_features, out_features=out_features, bias=True
            ),
        )

        self.predictor = nn.Sequential(
            nn.Linear(
                in_features=out_features, out_features=hidden_features, bias=False
            ),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_features, out_features=128, bias=True),
        )

        self.global_batch_size = (
            self.hparams.num_nodes * self.hparams.gpus * self.hparams.batch_size
            if self.hparams.gpus > 0
            else self.hparams.batch_size
        )
        self.train_iters_per_epoch = self.hparams.num_samples // self.global_batch_size
        self.total_steps = self.train_iters_per_epoch * self.hparams.max_epochs

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
        return self.left_model(x)

    def _regression_loss(self, x, y, epsilon: float = 1e-5):
        """Cosine-similarity based loss."""
        normed_x = x / torch.unsqueeze(torch.linalg.norm(x, dim=1, ord=2), axis=-1)
        normed_y = y / torch.unsqueeze(torch.linalg.norm(y, dim=1, ord=2), axis=-1)
        return torch.mean(0.5 * torch.sum((normed_x - normed_y) ** 2, axis=-1))

    def _safe_norm(self, x, min_norm: float):
        """Compute normalization, with correct gradients."""
        norm = torch.linalg.norm(x)
        x = torch.where(norm <= min_norm, torch.ones_like(x), x)
        return torch.where(norm <= min_norm, min_norm, torch.linalg.norm(x))

    def on_after_batch_transfer(self, batch, dataloader_idx):
        a, p, y = batch
        a = rearrange(a["spatial_sample"], "b t c w h -> b c t w h")
        p = rearrange(p["spatial_sample"], "b t c w h -> b c t w h")
        return a, p, y

    def forward_left(self, x):
        emb = self.left_model(x)
        proj = self.projector(emb)
        pred = self.predictor(proj)
        return pred

    def forward_right(self, x):
        emb = self.right_model(x)
        proj = self.projector(emb)
        pred = self.predictor(proj)
        return pred

    def forward_left2right(self, a, p):

        left_pred = self.forward_left(a)

        with torch.no_grad():
            right_pred = self.forward_right(p)

        loss = self._regression_loss(left_pred, right_pred)
        return loss

    def forward_right2left(self, a, p):

        right_pred = self.forward_right(p)
        with torch.no_grad():
            left_pred = self.forward_left(a)

        loss = self._regression_loss(left_pred, right_pred)
        return loss

    def shared_step(self, batch):
        a, p, y = batch
        left_loss = self.forward_left2right(a, p)
        right_loss = self.forward_right2left(a, p)

        loss = left_loss + right_loss
        return torch.mean(loss)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        if self.hparams.exclude_bn_bias:
            params = self.hparams.exclude_from_wt_decay(
                self.hparams.named_parameters(), weight_decay=self.hparams.weight_decay
            )
        else:
            params = self.parameters()

        optimizer = LARS(
            params,
            lr=self.hparams.learning_rate,
            momentum=0.9,
            weight_decay=self.hparams.weight_decay,
            trust_coefficient=0.001,
        )

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                linear_warmup_decay(
                    self.hparams.warmup_steps, self.total_steps, cosine=True
                ),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config)

    data_module = SupervisedPanAfPairsDataModule(cfg=cfg)

    model = ActionClassifier(
        gpus=cfg.getint("trainer", "gpus"),
        num_samples=17624,  # Need to auto-calculate this
        batch_size=cfg.getint("loader", "batch_size"),
        num_nodes=cfg.getint("trainer", "num_nodes"),
    )

    online_evaluator = SSLOnlineEvaluator(
        drop_p=0.0,
        hidden_dim=None,
        z_dim=2048,
        num_classes=9,
    )

    wand_logger = WandbLogger(offline=True)

    which_classes = cfg.get("dataset", "classes") if not NoOptionError else "all"
    per_class_acc_callback = PerClassAccuracy(which_classes=which_classes)

    online_evaluator = SSLOnlineEvaluator(
        z_dim=2048,
        hidden_dim=None,
        drop_p=0.0,
        num_classes=9,
    )

    val_top1_acc_checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/byol/val_top1_acc", monitor="val_top1_acc", mode="max"
    )

    val_per_class_acc_checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/byol/val_per_class_acc",
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
            callbacks=[online_evaluator],
            fast_dev_run=5,
        )
    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()
