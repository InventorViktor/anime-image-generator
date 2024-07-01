import os

import pytorch_lightning as pl
import torch
from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
from pytorch_lightning import LightningModule, Trainer


class SampleImagesCallBack(pl.Callback):
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if (trainer.current_epoch + 1) % 5 != 0:
            return

        self.save_images(trainer, pl_module)

    def save_images(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pipeline = DDPMPipeline(pl_module.model, pl_module.noise_scheduler)
        pipeline.set_progress_bar_config(disable=True)

        images = pipeline(
            batch_size=16,
            generator=torch.Generator(device=pl_module.device).manual_seed(42),
        ).images

        test_dir = "samples"
        os.makedirs(test_dir, exist_ok=True)

        epoch = trainer.current_epoch
        image_grid = make_image_grid(images, rows=4, cols=4)

        image_grid.save(f"{test_dir}/{epoch:04d}.png")
