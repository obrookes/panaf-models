import torch
from contextlib import contextmanager
from typing import Any, Dict, Optional, Sequence, Tuple, Union
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities import rank_zero_warn
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Optimizer
from pl_bolts.models.self_supervised.evaluator import SSLEvaluator


class SSLOnlineEvaluator(Callback):  # pragma: no cover
    """
    Attaches a MLP for fine-tuning using the standard self-supervised protocol.
    """

    def __init__(
        self,
        z_dim: int,
        drop_p: float = 0.2,
        hidden_dim: Optional[int] = None,
        num_classes: Optional[int] = None,
        strategy: Optional[str] = None,
    ):
        """
        Args:
            z_dim: Representation dimension
            drop_p: Dropout probability
            hidden_dim: Hidden dimension for the fine-tune MLP
        """
        super().__init__()

        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.drop_p = drop_p

        self.optimizer: Optional[Optimizer] = None
        self.online_evaluator: Optional[SSLEvaluator] = None
        self.num_classes: Optional[int] = num_classes
        self.strategy: Optional[str] = strategy

        self._recovered_callback_state: Optional[Dict[str, Any]] = None

    def setup(
        self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None
    ) -> None:
        if self.num_classes is None:
            self.num_classes = trainer.datamodule.num_classes

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # must move to device after setup, as during setup, pl_module is still on cpu
        self.online_evaluator = SSLEvaluator(
            n_input=self.z_dim,
            n_classes=self.num_classes,
            p=self.drop_p,
            n_hidden=self.hidden_dim,
        ).to(pl_module.device)

        if self.strategy == "ddp":
            from torch.nn.parallel import DistributedDataParallel as DDP

            self.online_evaluator = DDP(
                self.online_evaluator, device_ids=[pl_module.device]
            )

        self.optimizer = torch.optim.Adam(self.online_evaluator.parameters(), lr=1e-4)

        if self._recovered_callback_state is not None:
            self.online_evaluator.load_state_dict(
                self._recovered_callback_state["state_dict"]
            )
            self.optimizer.load_state_dict(
                self._recovered_callback_state["optimizer_state"]
            )

    def to_device(
        self, batch: Sequence, device: Union[str, torch.device]
    ) -> Tuple[Tensor, Tensor]:
        # get the labeled batch

        _, _, x, y = batch

        # last input is for online eval
        x = x.to(device)
        y = y.to(device)

        return x, y

    def shared_step(
        self,
        pl_module: LightningModule,
        batch: Sequence,
    ):
        with torch.no_grad():
            with set_training(pl_module, False):
                x, y = self.to_device(batch, pl_module.device)
                representations = pl_module(x).flatten(start_dim=1)

        # forward pass
        mlp_logits = self.online_evaluator(representations)  # type: ignore[operator]
        mlp_loss = F.cross_entropy(mlp_logits, y)

        return mlp_logits, mlp_loss, y

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:

        mlp_logits, mlp_loss, y = self.shared_step(pl_module, batch)

        pl_module.top1_train_accuracy(mlp_logits.softmax(-1), y)
        pl_module.train_per_class_accuracy(mlp_logits.softmax(-1), y)

        # update finetune weights
        mlp_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        pl_module.log("mlp_loss", mlp_loss, on_step=True, on_epoch=False, prog_bar=True)

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.log("train_acc", pl_module.top1_train_accuracy, prog_bar=True)

        pl_module.log(
            "train_per_class_acc", pl_module.train_per_class_accuracy, prog_bar=True
        )

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:

        mlp_logits, mlp_loss, y = self.shared_step(pl_module, batch)

        pl_module.top1_val_accuracy(mlp_logits.softmax(-1), y)
        pl_module.val_per_class_accuracy(mlp_logits.softmax(-1), y)

        pl_module.log(
            "online_val_loss",
            mlp_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def on_validation_epoch_end(self, trainer, pl_module):

        pl_module.log("val_top1_acc_epoch", pl_module.top1_val_accuracy, prog_bar=True)

        pl_module.log(
            "val_per_class_acc", pl_module.val_per_class_accuracy, prog_bar=True
        )

    def on_save_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]
    ) -> dict:
        return {
            "state_dict": self.online_evaluator.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }

    def on_load_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        callback_state: Dict[str, Any],
    ) -> None:
        self._recovered_callback_state = callback_state


@contextmanager
def set_training(module: nn.Module, mode: bool):
    """Context manager to set training mode.

    When exit, recover the original training mode.
    Args:
        module: module to set training mode
        mode: whether to set training mode (True) or evaluation mode (False).
    """
    original_mode = module.training

    try:
        module.train(mode)
        yield module
    finally:
        module.train(original_mode)
