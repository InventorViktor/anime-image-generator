import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from anime_image_generator.callbacks.SampleImages import SampleImagesCallBack
from anime_image_generator.datamodules.image_datamodule import ImageDataModule
from anime_image_generator.models.ImageDiffuser import ImageDiffuser

data_module = ImageDataModule("E:\\PythonProjects\\anime-images\\images")
model = ImageDiffuser()

logger = TensorBoardLogger("lightning_logs", name="image_diffuser")
trainer = pl.Trainer(
    accelerator="gpu",
    logger=logger,
    max_epochs=30,
    callbacks=[SampleImagesCallBack()],
    gradient_clip_val=1.0,
)

trainer.fit(model, data_module)
