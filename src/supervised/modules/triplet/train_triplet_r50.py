import torch
import argparse
import configparser
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from panaf.datamodules import SupervisedPanAfDataModule
from sklearn.neighbors import KNeighborsClassifier
from src.supervised.utils.model_initialiser import initialise_triplet_model
from src.supervised.utils.miner_initialiser import initialise_miner
from src.supervised.utils.loss_initialiser import initialise_loss
from src.supervised.callbacks.custom_metrics import PerClassAccuracy
from configparser import NoOptionError

# from losses import OnlineReciprocalTripletLoss


class ActionClassifier(pl.LightningModule):
    def __init__(
        self,
        lr,
        weight_decay,
        model_name,
        freeze_backbone,
        embedding_size,
        num_classes,
        type_of_fusion,
        miner_name,
        margin,
        type_of_triplets,
        loss_name,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = initialise_triplet_model(
            name=model_name,
            freeze_backbone=freeze_backbone,
            embedding_size=embedding_size,
            type_of_fusion=type_of_fusion,
        )

        self.classifier = KNeighborsClassifier(n_neighbors=9)

        self.triplet_miner = initialise_miner(
            name=miner_name, margin=margin, type_of_triplets=type_of_triplets
        )

        self.triplet_loss = initialise_loss(name=loss_name)

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
        emb, pred = self.model(x)
        return emb, pred

    def training_step(self, batch, batch_idx):

        x, y = batch
        embeddings, preds = self(x)

        self.train_top1_acc(preds, y)
        self.train_avg_per_class_acc(preds, y)
        self.train_per_class_acc.update(preds, y)

        miner_output = self.triplet_miner(embeddings, y)
        loss = self.triplet_loss(embeddings, y, miner_output)

        return {"loss": loss}

    def training_epoch_end(self, outputs):

        # Log epoch acc
        self.log(
            "train_top1_acc",
            self.train_top1_acc,
            logger=True,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

        # Log epoch acc
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
        embeddings, preds = self(x)

        self.val_top1_acc(preds, y)
        self.val_avg_per_class_acc(preds, y)
        self.val_per_class_acc.update(preds, y)

        miner_output = self.triplet_miner(embeddings, y)
        loss = self.triplet_loss(embeddings, y, miner_output)
        return {"loss": loss}

    def _reciprocal_triplet_loss(self, embeddings, y):
        a_idx, p_idx, n_idx = self.triplet_miner(embeddings, y)
        labels = torch.cat((y[a_idx], y[p_idx], y[n_idx]), dim=0)

        triplet_loss = self.triplet_loss(
            embeddings[a_idx],
            embeddings[p_idx],
            embeddings[n_idx],
            labels,
        )
        return triplet_loss

    def _ce_loss(self, preds, y):
        ce_loss = self.ce_loss(preds, y)
        return ce_loss

    def _hybrid_loss(triplet_loss, ce_loss, tau=0.01):
        return tau * triplet_loss + ce_loss

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

    model_name = cfg.get("model", "name")
    embedding_size = cfg.getint("model", "embedding_size")
    type_of_fusion = cfg.get("model", "type_of_fusion")

    data_type = cfg.get("dataset", "type")
    miner_name = cfg.get("triplets", "miner")
    margin = cfg.getfloat("triplets", "margin")
    type_of_triplets = cfg.get("triplets", "type_of_triplets")
    loss_name = cfg.get("triplets", "loss")

    model = ActionClassifier(
        lr=cfg.getfloat("hparams", "lr"),
        weight_decay=cfg.getfloat("hparams", "weight_decay"),
        model_name=cfg.get("dataset", "type"),
        freeze_backbone=cfg.getboolean("hparams", "freeze_backbone"),
        embedding_size=embedding_size,
        type_of_fusion=type_of_fusion,
        num_classes=9,
        miner_name=miner_name,
        margin=margin,
        type_of_triplets=type_of_triplets,
        loss_name=loss_name,
    )
    wand_logger = WandbLogger(offline=True)

    which_classes = cfg.get("dataset", "classes") if not NoOptionError else "all"
    per_class_acc_callback = PerClassAccuracy(which_classes=which_classes)

    val_top1_acc_checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/val_top1_acc/type={data_type}_margin={margin}_triplets={type_of_triplets}",
        monitor="val_top1_acc",
        mode="max",
    )

    val_per_class_acc_checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/val_per_class_acc/type={data_type}_margin={margin}_triplets={type_of_triplets}",
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
            strategy=cfg.get("trainer", "strategy"),
            max_epochs=cfg.getint("trainer", "max_epochs"),
            stochastic_weight_avg=cfg.getboolean("trainer", "swa"),
            fast_dev_run=2,
        )
    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()
