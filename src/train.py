import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from anime_image_generator.callbacks.SampleImages import SampleImagesCallBack
from anime_image_generator.datamodules.image_datamodule import ImageDataModule
from anime_image_generator.models.ImageDiffuser import ImageDiffuser

data_module = ImageDataModule("path/to/image/folder")
model = ImageDiffuser()

logger = TensorBoardLogger("lightning_logs", name="image_diffuser")
trainer = pl.Trainer(
    accelerator="gpu",
    logger=logger,
    max_epochs=30,
    precision="16-mixed",
    # fast_dev_run=True,
    callbacks=[SampleImagesCallBack()],
)

trainer.fit(model, data_module)
