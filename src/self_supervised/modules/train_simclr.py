import math
import torch
import argparse
import torchvision
import configparser
import torchmetrics
import pytorch_lightning as pl
from torch import Tensor
from einops import rearrange
from kornia import tensor_to_image
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from panaf.datamodules import SupervisedPanAfDataModule
from src.self_supervised.augmentations.simclr_augs import (
    SimCLRTrainDataTransform,
    SimCLREvalDataTransform,
)

from src.self_supervised.models.resnets import ResNet50
from src.self_supervised.models.mlp import MLP
from pl_bolts.optimizers import LARS
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from src.self_supervised.callbacks.callback import SSLOnlineEvaluator
import matplotlib.pyplot as plt


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
        hidden_mlp: int = 2048,
        feat_dim: int = 128,
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
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()

        self.encoder = ResNet50()
        self.projection = MLP()

        self.mlp = MLP()

        global_batch_size = (
            self.hparams.num_nodes * self.hparams.gpus * self.hparams.batch_size
            if self.hparams.gpus > 0
            else self.hparams.batch_size
        )
        self.train_iters_per_epoch = self.hparams.num_samples // global_batch_size

        # Instantiate augmentations
        self.train_augmentations = SimCLRTrainDataTransform()
        self.val_augmentations = SimCLREvalDataTransform()

        # Training metrics
        self.top1_train_accuracy = torchmetrics.Accuracy(top_k=1)
        self.train_per_class_accuracy = torchmetrics.Accuracy(
            num_classes=9, average="macro"
        )
        # Validation metrics
        self.top1_val_accuracy = torchmetrics.Accuracy(top_k=1)
        self.val_per_class_accuracy = torchmetrics.Accuracy(
            num_classes=9, average="macro"
        )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        x, y = batch
        x = x["spatial_sample"]

        x_r = rearrange(x, "b t c w h -> (b t) c w h")

        if self.trainer.training:
            x1, x2 = self.train_augmentations(x_r)
        if self.trainer.validating:
            x1, x2 = self.val_augmentations(x_r)

        x = rearrange(x, "b t c w h -> b c t w h")

        return x1, x2, x, y

    def forward(self, x):
        return self.encoder(x)

    def shared_step(self, batch):
        x1, x2, x, y = batch

        h1 = self.encoder(x1)
        h2 = self.encoder(x2)

        z1 = self.projection(h1)
        z2 = self.projection(h2)

        loss = self.nt_xent_loss(z1, z2, self.hparams.temperature)

        return loss

    def training_step(self, batch, batch_idx):

        # self.show_batch(batch=batch)
        loss = self.shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def exclude_from_wt_decay(
        self, named_params, weight_decay, skip_list=("bias", "bn")
    ):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": weight_decay},
            {
                "params": excluded_params,
                "weight_decay": 0.0,
            },
        ]

    def configure_optimizers(self):
        if self.hparams.exclude_bn_bias:
            params = self.hparams.exclude_from_wt_decay(
                self.hparams.named_parameters(), weight_decay=self.hparams.weight_decay
            )
        else:
            params = self.parameters()

        if self.hparams.optimizer == "lars":
            optimizer = LARS(
                params,
                lr=self.hparams.learning_rate,
                momentum=0.9,
                weight_decay=self.hparams.weight_decay,
                trust_coefficient=0.001,
            )
        elif self.hparams.optimizer == "adam":
            optimizer = torch.optim.Adam(
                params,
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )

        warmup_steps = self.train_iters_per_epoch * self.hparams.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.hparams.max_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps, total_steps, cosine=True),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def nt_xent_loss(self, out_1, out_2, temperature, eps=1e-6):
        """
        assume out_1 and out_2 are normalized
        out_1: [batch_size, dim]
        out_2: [batch_size, dim]
        """
        # gather representations in case of distributed training
        # out_1_dist: [batch_size * world_size, dim]
        # out_2_dist: [batch_size * world_size, dim]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            out_1_dist = SyncFunction.apply(out_1)
            out_2_dist = SyncFunction.apply(out_2)
        else:
            out_1_dist = out_1
            out_2_dist = out_2

        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = torch.mm(out, out_dist.t().contiguous())
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
        row_sub = Tensor(neg.shape).fill_(math.e ** (1 / temperature)).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()

        return loss

    def show_batch(self, batch, win_size=(15, 15)):
        def _to_vis(data):
            return tensor_to_image(torchvision.utils.make_grid(data, nrow=5))

        x_i, x_j = batch
        x_i = rearrange(x_i, "b c t w h -> (b t) c w h")
        x_j = rearrange(x_j, "b c t w h -> (b t) c w h")
        # get a batch from the training set: try with `val_datlaoader` :)
        # use matplotlib to visualize
        plt.figure(figsize=win_size)
        plt.imshow(_to_vis(x_i))
        plt.figure(figsize=win_size)
        plt.imshow(_to_vis(x_j))
        plt.show()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config)

    data_module = SupervisedPanAfDataModule(cfg=cfg)

    model = ActionClassifier(gpus=0, num_samples=17624, batch_size=2, num_nodes=1)

    online_evaluator = SSLOnlineEvaluator(
        drop_p=0.0,
        hidden_dim=None,
        z_dim=2048,
        num_classes=9,
        dataset=None,
    )

    wand_logger = WandbLogger(offline=True)

    val_top1_acc_checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/val_top1_acc", monitor="val_top1_acc_epoch", mode="max"
    )

    val_per_class_acc_checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/val_per_class_acc",
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
