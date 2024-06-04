import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, image_folder: str, transform: transforms = None):
        self.transform = transform
        self.img_names = os.listdir(image_folder)
        self.image_paths = [
            os.path.join(image_folder, img_name) for img_name in self.img_names
        ]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Image:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image
