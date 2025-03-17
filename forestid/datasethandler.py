from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch import tensor, long
from PIL import Image


class PlantDataset(Dataset):
    def __init__(self, image_paths, labels, output_size: tuple):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = self._create_transform(output_size)
        self.class_to_idx = {
            class_name: idx for idx, class_name in enumerate(sorted(set(labels)))
        }

    def get_class_idx(self, class_name: str):
        return self.class_to_idx[class_name]

    def get_class_name(self, idx: int):
        return [key for key, val in self.class_to_idx.items() if val == idx][0]

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
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        return transform
