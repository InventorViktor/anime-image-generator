import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from anime_image_generator.callbacks.SampleImages import SampleImagesCallBack
from anime_image_generator.datamodules.image_datamodule import ImageDataModule
from anime_image_generator.models.ImageDiffuser import ImageDiffuser

torch.set_float32_matmul_precision("medium")

data_module = ImageDataModule("path/to/image/folder")
model = ImageDiffuser()

logger = TensorBoardLogger("lightning_logs", name="image_diffuser")
trainer = pl.Trainer(
    accelerator="gpu",
    strategy="ddp",
    logger=logger,
    max_epochs=30,
    precision="16-mixed",
    callbacks=[SampleImagesCallBack()],
    accumulate_grad_batches=4,
)

trainer.fit(model, data_module)
