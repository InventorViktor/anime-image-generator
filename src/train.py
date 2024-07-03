from anime_image_generator.datamodules.image_datamodule import ImageDataModule
from anime_image_generator.models.ImageDiffuser import ImageDiffuser

dataloader = ImageDataModule("path/to/images").init_loader()
model = ImageDiffuser(dataloader)

model.fit()
