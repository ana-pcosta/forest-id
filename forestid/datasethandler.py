from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch import tensor, long
from PIL import Image
import json


class PlantDataset(Dataset):
    def __init__(
        self,
        image_paths,
        labels,
        output_size: tuple,
        classes_names=None,
        class_id_mapping=None,
        augment: bool = False,
    ):
        self.image_paths = image_paths
        self.labels = labels
        if augment:
            self.transform = transforms.Compose(
                [
                    *self._create_augmentation().transforms,
                    *self._create_transform(output_size).transforms,
                ]
            )
        else:
            self.transform = self._create_transform(output_size)

        if class_id_mapping is not None:
            self.class_to_idx = class_id_mapping
        elif classes_names is not None:
            self.class_to_idx = {
                class_name: idx
                for idx, class_name in enumerate(sorted(set(classes_names)))
            }
        else:
            raise ValueError(
                "Either class_id_mapping or classes_names must be provided"
            )

    def get_class_idx(self, class_name: str):
        return self.class_to_idx[class_name]

    def get_class_name(self, idx: int):
        return [key for key, val in self.class_to_idx.items() if val == idx][0]

    def save_class_mapping(self, save_path: str):
        with open(save_path, "w") as f:
            json.dump(self.class_to_idx, f)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.class_to_idx[self.labels[idx]]
        # Convert label from string to integer index
        label = tensor(label, dtype=long)

        if self.transform:
            image = self.transform(image)

        return image, label

    def _create_transform(self, output_size: tuple):
        transform = transforms.Compose(
            [
                transforms.Resize(output_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        return transform

    def _create_augmentation(self):
        augment = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            ]
        )
        return augment
