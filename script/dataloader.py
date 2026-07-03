import json
import os
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ImageNet(Dataset):
    def __init__(
        self,
        img_dir: str,
        annotations_file: str,
        transform: Optional[Callable] = None,
    ):
        self.img_dir = img_dir
        self.transform = transform
        self.samples: List[Tuple[str, int, str, str]] = []
        self.class_id_to_name: Dict[int, str] = {}
        self.all_class_names: List[str] = []
        self._load_annotations(annotations_file)

    def _load_annotations(self, annotations_file: str) -> None:
        with open(annotations_file, "r", encoding="utf-8") as f:
            annotations = json.load(f)  # {folder_name: [class_id, class_name]}

        for folder_name, value in annotations.items():
            class_id, class_name = value
            class_id = int(class_id)
            class_name = str(class_name).replace("_", " ")
            self.class_id_to_name[class_id] = class_name

            folder_path = os.path.join(self.img_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            for file_name in os.listdir(folder_path):
                img_path = os.path.abspath(os.path.join(folder_path, file_name))
                if not os.path.isfile(img_path):
                    continue
                self.samples.append((img_path, class_id, class_name, folder_name))

        if self.class_id_to_name:
            max_class_id = max(self.class_id_to_name.keys())
            self.all_class_names = ["" for _ in range(max_class_id + 1)]
            for class_id, class_name in self.class_id_to_name.items():
                self.all_class_names[class_id] = class_name

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, class_id, class_name, folder_name = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, class_id, class_name, img_path, folder_name


def get_imagenet_dataloader(
    img_dir: str,
    annotations_file: str,
    batch_size: int,
    transform: Callable,
    num_workers: int = 4,
    shuffle: bool = False,
) -> Tuple[DataLoader, List[str]]:
    dataset = ImageNet(img_dir=img_dir, annotations_file=annotations_file, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, dataset.all_class_names




        
    
        
    
        
        
        