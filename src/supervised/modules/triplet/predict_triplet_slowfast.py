import torch
import argparse
import configparser
import torchmetrics
import pytorch_lightning as pl
import numpy as np
from torch import nn
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from panaf.datamodules import SupervisedPanAfDataModule
from pytorch_metric_learning.miners import TripletMarginMiner
from losses import OnlineReciprocalTripletLoss
from sklearn.neighbors import KNeighborsClassifier
from src.supervised.utils.model_initialiser import initialise_triplet_model
from src.supervised.callbacks.custom_metrics import PerClassAccuracy
from configparser import NoOptionError


class ActionClassifier(pl.LightningModule):
    def __init__(
        self, lr, weight_decay, model_name, freeze_backbone, margin, type_of_triplets
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = initialise_triplet_model(
            name=model_name, freeze_backbone=freeze_backbone
        )

        self.classifier = KNeighborsClassifier(n_neighbors=9)

        self.triplet_miner = TripletMarginMiner(
            margin=margin, type_of_triplets=type_of_triplets
        )
        self.triplet_loss = OnlineReciprocalTripletLoss()  # self.selector
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

        # Test metrics
        self.test_top1_acc = torchmetrics.Accuracy(top_k=1)
        self.test_avg_per_class_acc = torchmetrics.Accuracy(
            num_classes=9, average="macro"
        )
        self.test_per_class_acc = torchmetrics.Accuracy(num_classes=9, average="none")

    def assign_embedding_name(self, name):
        self.embedding_filename = name

    def forward(self, x):
        emb, pred = self.model(x)
        return emb, pred

    def shared_step(self, batch):
        x, y = batch
        slow = x["spatial_sample"].permute(0, 2, 1, 3, 4)
        fast = self.uniform_temporal_subsample(x=slow, temporal_dim=2, num_samples=8)
        embeddings, preds = self.model([fast, slow])
        return embeddings, preds, y

    def training_step(self, batch, batch_idx):

        embeddings, preds, y = self.shared_step(batch)

        self.train_top1_acc(preds, y)
        self.train_avg_per_class_acc(preds, y)
        self.train_per_class_acc.update(preds, y)

        a_idx, p_idx, n_idx = self.triplet_miner(embeddings, y)
        labels = torch.cat((y[a_idx], y[p_idx], y[n_idx]), dim=0)

        triplet_loss = self.triplet_loss(
            embeddings[a_idx],
            embeddings[p_idx],
            embeddings[n_idx],
            labels,
        )
        ce_loss = self.ce_loss(preds, y)
        loss = 0.01 * triplet_loss + ce_loss

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

        embeddings, preds, y = self.shared_step(batch)

        self.val_top1_acc(preds, y)
        self.val_avg_per_class_acc(preds, y)
        self.val_per_class_acc.update(preds, y)

        a_idx, p_idx, n_idx = self.triplet_miner(embeddings, y)
        labels = torch.cat((y[a_idx], y[p_idx], y[n_idx]), dim=0)

        triplet_loss = self.triplet_loss(
            embeddings[a_idx],
            embeddings[p_idx],
            embeddings[n_idx],
            labels,
        )
        ce_loss = self.ce_loss(preds, y)
        loss = 0.01 * triplet_loss + ce_loss

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

    def test_step(self, batch, batch_idx):

        embeddings, preds, y = self.shared_step(batch)

        self.test_top1_acc(preds, y)
        self.test_avg_per_class_acc(preds, y)
        self.test_per_class_acc.update(preds, y)

        a_idx, p_idx, n_idx = self.triplet_miner(embeddings, y)
        labels = torch.cat((y[a_idx], y[p_idx], y[n_idx]), dim=0)

        triplet_loss = self.triplet_loss(
            embeddings[a_idx],
            embeddings[p_idx],
            embeddings[n_idx],
            labels,
        )
        ce_loss = self.ce_loss(preds, y)
        loss = 0.01 * triplet_loss + ce_loss

        return {"loss": loss}

    def test_epoch_end(self, outputs):

        # Log top-1 acc per epoch
        self.log(
            "test_top1_acc",
            self.test_top1_acc,
            logger=True,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

        # Log per class acc per epoch
        self.log(
            "test_avg_per_class_acc",
            self.test_avg_per_class_acc,
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

    def uniform_temporal_subsample(self, x, temporal_dim, num_samples):
        t = x.shape[temporal_dim]
        assert num_samples > 0 and t > 0
        # Sample by nearest neighbor interpolation if num_samples > t.
        indices = torch.linspace(0, t - 1, num_samples, device=self.device)
        indices = torch.clamp(indices, 0, t - 1).long()
        fast = torch.index_select(x, temporal_dim, indices)
        return fast

    def on_predict_epoch_start(self):

        # Embeddings/labels to be stored on the inference set
        self.outputs_embedding = np.zeros((1, 128))
        self.labels_embedding = np.zeros((1))

    def predict_step(self, batch, batch_idx):
        embeddings, preds, y = self.shared_step(batch)
        self.outputs_embedding = np.concatenate(
            (self.outputs_embedding, embeddings.detach().cpu()), axis=0
        )
        self.labels_embedding = np.concatenate(
            (self.labels_embedding, y.detach().cpu()), axis=0
        )

    def on_predict_epoch_end(self, results):
        np.savez(
            self.embedding_filename,
            embeddings=self.outputs_embedding,
            labels=self.labels_embedding,
        )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    args = parser.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config)

    data_module = SupervisedPanAfDataModule(cfg=cfg)
    model = ActionClassifier.load_from_checkpoint(cfg.get("trainer", "ckpt"))

    name = f"{args.prefix}_{args.split}_embeddings.npz"
    model.assign_embedding_name(name)

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
            logger=wand_logger,
            fast_dev_run=5,
        )

    data_module.setup()

    if args.split == "train":
        loader = data_module.train_dataloader()
    elif args.split == "validation":
        loader = data_module.val_dataloader()
        trainer.validate(model, dataloaders=loader)
        trainer.predict(model, dataloaders=loader)
    elif args.split == "test":
        loader = data_module.test_dataloader()
        trainer.test(model, dataloaders=loader)
        trainer.predict(model, dataloaders=loader)





if __name__ == "__main__":
    main()
