import os

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DDPMPipeline, DDPMScheduler
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class ImageDiffuser:
    def __init__(self, dataloader: DataLoader) -> None:
        super().__init__()
        self.dataloader = dataloader
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
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    def fit(self) -> None:
        dataset_size = 100_000
        num_epochs = 30

        num_training_steps = dataset_size * num_epochs
        num_warmup_steps = int(0.01 * num_training_steps)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        accelerator = Accelerator(
            mixed_precision="fp16",
            gradient_accumulation_steps=1,
            log_with="tensorboard",
            project_dir=os.path.join("ddpm-butterflies-128", "logs"),
        )

        model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            self.model, optimizer, self.dataloader, lr_scheduler
        )

        global_step = 0

        for epoch in range(30):
            progress_bar = tqdm(
                total=len(dataloader), disable=not accelerator.is_local_main_process
            )
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(dataloader):
                clean_images = batch

                noise = torch.randn(clean_images.shape, device=clean_images.device)
                bs = clean_images.shape[0]

                timesteps = torch.randint(
                    0,
                    self.noise_scheduler.config.num_train_timesteps,
                    (bs,),
                    device=clean_images.device,
                    dtype=torch.int64,
                )

                noisy_images = self.noise_scheduler.add_noise(
                    clean_images, noise, timesteps
                )

                with accelerator.accumulate(model):
                    noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                    loss = F.mse_loss(noise_pred, noise)
                    accelerator.backward(loss)

                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                progress_bar.update(1)
                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step,
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

            if epoch == 0 or epoch % 5 == 0:
                self.save_images(epoch, accelerator)

            if epoch == 0 or epoch == 29:
                pipeline = DDPMPipeline(
                    unet=accelerator.unwrap_model(self.model),
                    scheduler=self.noise_scheduler,
                )
                out_dir = "pipeline"
                os.makedirs(out_dir, exist_ok=True)
                pipeline.save_pretrained(out_dir)

    def save_images(self, epoch: int, accelerator: Accelerator) -> None:
        pipeline = DDPMPipeline(
            unet=accelerator.unwrap_model(self.model), scheduler=self.noise_scheduler
        )

        images = pipeline(
            batch_size=16,
            generator=torch.Generator(device="cuda").manual_seed(0),
        ).images

        test_dir = "samples"
        os.makedirs(test_dir, exist_ok=True)

        image_grid = make_image_grid(images, rows=4, cols=4)
        image_grid.save(f"{test_dir}/{epoch:04d}.png")
