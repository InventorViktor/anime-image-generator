from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import Tensor


class ImageDiffuser(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = UNet2DModel(
            sample_size=64,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        self.noise_scheduler = DDPMScheduler(1000)

    def training_step(self, batch: Any, batch_idx: Any) -> Tensor:
        noise = torch.rand(batch.shape, device=self.device)
        bs = batch.shape[0]

        time_steps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps, (bs,), device=self.device
        ).long()

        noisy_images = self.noise_scheduler.add_noise(batch, noise, time_steps)
        noise_pred = self.model(noisy_images, time_steps).sample

        loss = F.mse_loss(noise_pred, noise)
        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

        dataset_size = 100_000
        batch_size = 64
        num_epochs = 30
        num_training_steps = (dataset_size // batch_size) * num_epochs
        num_warmup_steps = int(0.1 * num_training_steps)

        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
