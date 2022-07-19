import copy
import torch
import argparse
import configparser
import torchmetrics
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from typing import Callable
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from panaf.datamodules import SupervisedPanAfDataModule
from src.self_supervised.augmentations.simclr_augs import (
    SimCLRTrainDataTransform,
    SimCLREvalDataTransform,
)
from configparser import NoOptionError
from src.self_supervised.callbacks.custom_metrics import PerClassAccuracy
from src.self_supervised.models.resnets import ResNet50
from src.self_supervised.callbacks.ssl import SSLOnlineEvaluator


class SyncFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [
            torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())
        ]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        # possible change: grad_input.contiguous()
        torch.distributed.all_reduce(
            grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False
        )

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]


class ActionClassifier(pl.LightningModule):
    def __init__(
        self,
        gpus: int,
        num_samples: int,
        batch_size: int,
        num_nodes: int = 1,
        arch: str = "resnet50",
        feature_dim: int = 2048,
        hidden_mlp: int = 4096,
        feat_dim: int = 256,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        temperature: float = 0.1,
        first_conv: bool = True,
        maxpool1: bool = True,
        optimizer: str = "adam",
        exclude_bn_bias: bool = False,
        start_lr: float = 0.0,
        learning_rate: float = 1e-3,
        final_lr: float = 0.0,
        weight_decay: float = 1e-6,
        mmt: float = 0.99,
        norm: Callable = nn.SyncBatchNorm,
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()

        self.mmt = mmt
        self.feature_dim = feature_dim

        backbone = ResNet50()
        projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_mlp, bias=False),
            norm(hidden_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_mlp, feat_dim, bias=True),
        )

        if projector is not None:
            backbone = nn.Sequential(
                backbone,
                projector,
            )

        self.backbone = backbone
        self.backbone_mmt = copy.deepcopy(backbone)

        for p in self.backbone_mmt.parameters():
            p.requires_grad = False

        self.predictor = nn.Sequential(
            nn.Linear(feat_dim, hidden_mlp, bias=False),
            norm(hidden_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_mlp, feat_dim, bias=True),
        )

        # Instantiate augmentations
        self.train_augmentations = SimCLRTrainDataTransform()
        self.val_augmentations = SimCLREvalDataTransform()

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

    def sim_loss(self, q, k):
        """
        Similarity loss for byol.
        Args:
            q and k (nn.tensor): inputs to calculate the similarity, expected to have
                the same shape of `N x C`.
        """
        similarity = torch.einsum("nc,nc->n", [q, k])
        loss = -similarity.mean()
        return loss

    def update_mmt(self, mmt: float):
        """
        Update the momentum. This function can be used to perform momentum annealing.
        Args:
            mmt (float): update the momentum.
        """
        self.mmt = mmt

    def get_mmt(self) -> float:
        """
        Get the momentum. This function can be used to perform momentum annealing.
        """
        return self.mmt

    @torch.no_grad()
    def _momentum_update_backbone(self):
        """
        Momentum update on the backbone.
        """
        for param, param_mmt in zip(
            self.backbone.parameters(), self.backbone_mmt.parameters()
        ):
            param_mmt.data = param_mmt.data * self.mmt + param.data * (1.0 - self.mmt)

    @torch.no_grad()
    def forward_backbone_mmt(self, x):
        """
        Forward momentum backbone.
        Args:
            x (tensor): input to be forwarded.
        """
        with torch.no_grad():
            proj = self.backbone_mmt(x)
        return F.normalize(proj, dim=1)

    def forward_backbone(self, x):
        """
        Forward backbone.
        Args:
            x (tensor): input to be forwarded.
        """
        proj = self.backbone(x)
        pred = self.predictor(proj)
        return F.normalize(pred, dim=1)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        x, y = batch
        x = x["spatial_sample"]

        if self.trainer.training or self.trainer.sanity_checking:
            x1, x2 = self.train_augmentations(x)
        if self.trainer.validating:
            x1, x2 = self.val_augmentations(x)

        x = rearrange(x, "b t c w h -> b c t w h")

        return x1, x2, x, y

    def forward(self, x):
        return self.backbone[0](x)

    def shared_step(self, batch):
        x1, x2, x, y = batch

        pred_1 = self.forward_backbone(x1)
        pred_2 = self.forward_backbone(x2)

        with torch.no_grad():
            self._momentum_update_backbone()
            proj_mmt_1 = self.forward_backbone_mmt(x1)
            proj_mmt_2 = self.forward_backbone_mmt(x2)

        loss = (
            self.sim_loss(pred_1, proj_mmt_2) + self.sim_loss(pred_2, proj_mmt_1)
        ) / 2
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
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
                callbacks=[
                    val_top1_acc_checkpoint_callback,
                    val_per_class_acc_checkpoint_callback,
                    online_evaluator,
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
                callbacks=[online_evaluator],
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
            callbacks=[per_class_acc_callback],
            fast_dev_run=5,
        )
    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()
