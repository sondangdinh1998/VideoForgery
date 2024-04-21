from typing import Any

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from pytorch_lightning.utilities.types import (
    STEP_OUTPUT,
    EVAL_DATALOADERS,
    TRAIN_DATALOADERS,
)

from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy_with_logits
from torchmetrics.classification import BinaryF1Score

from models.recognition import DeepfakeDetection
from optims.optimizer import build_optimizer
from optims.scheduler import get_lr, set_lr


class DeepfakeDetectionTask(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.model = DeepfakeDetection(self.hparams.model.detection)
        self.metric = BinaryF1Score()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        dataset = instantiate(self.hparams.dataset.trainset, _recursive_=False)
        data_dl = DataLoader(
            dataset,
            **self.hparams.dataset.dataloader,
            shuffle=True,
        )
        return data_dl

    def val_dataloader(self) -> EVAL_DATALOADERS:
        dataset = instantiate(self.hparams.dataset.trainset, _recursive_=False)
        data_dl = DataLoader(
            dataset,
            **self.hparams.dataset.dataloader,
            shuffle=False,
        )
        return data_dl

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        xs, ys = batch

        logits = self.model([xs.transpose(1, 2)]).squeeze(1)
        targets = ys.float()

        loss = binary_cross_entropy_with_logits(logits, targets)
        self.log("train_loss", loss, sync_dist=True, prog_bar=True)

        temporal_optim, spatial_optim = self.optimizers()
        temporal_weight, spatial_weight = self.hparams.model.weight.values()

        if batch_idx % (temporal_weight + spatial_weight) < temporal_weight:
            temporal_optim.zero_grad()
            self.manual_backward(loss)

            learning_rate = get_lr(self.hparams.optimizer, self.current_epoch)
            set_lr(temporal_optim, learning_rate)

            temporal_optim.step()
        else:
            spatial_optim.zero_grad()
            self.manual_backward(loss)

            learning_rate = get_lr(self.hparams.optimizer, self.current_epoch)
            set_lr(spatial_optim, learning_rate)

            spatial_optim.step()

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        xs, ys = batch

        logits = self.model([xs.transpose(1, 2)]).squeeze(1)
        targets = ys.float()

        loss = binary_cross_entropy_with_logits(logits, targets)
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)

        self.metric(logits.sigmoid(), targets)
        self.log("val_score", self.metric, sync_dist=True)

    def configure_optimizers(self) -> Any:
        temporal_optim, spatial_optim = build_optimizer(
            self.model, self.hparams.optimizer
        )
        return [temporal_optim, spatial_optim]


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config):
    task = DeepfakeDetectionTask(**config.task)
    callbacks = [instantiate(cfg) for _, cfg in config.callbacks.items()]
    trainer = pl.Trainer(**config.trainer, callbacks=callbacks)
    trainer.fit(task)


if __name__ == "__main__":
    main()
