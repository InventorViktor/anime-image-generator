import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms

from anime_image_generator.datasets.image_dataset import ImageDataset


class ImageDataModule(pl.LightningDataModule):
    def __init__(self, image_folder: str, batch_size: int = 32):
        super().__init__()
        self.image_folder = image_folder
        self.batch_size = batch_size

    def setup(self, stage: str) -> None:
        transform = transforms.Compose(
            [
                transforms.Resize(128),
                transforms.CenterCrop((128, 128)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.train_dataset = ImageDataset(
            image_folder=self.image_folder, transform=transform
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
