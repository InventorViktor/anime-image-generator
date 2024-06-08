from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from diffusers.models.unets.unet_2d import UNet2DModel
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import Tensor


class ImageDiffuser(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = UNet2DModel(
            sample_size=384,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",  # 6x6
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        self.noise_scheduler = DDPMScheduler(4000)

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
        return torch.optim.AdamW(self.model.parameters(), lr=4e-4)
