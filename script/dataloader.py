import json
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ImageNet(Dataset):
    def __init__(
        self,
        img_dir: str,
        annotations_file: str,
        transform: Optional[Callable] = None,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
    ):
        self.img_dir = img_dir
        self.transform = transform
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.annotations_file = annotations_file
        self.samples: List[Tuple[str, int, str, str]] = []
        self.class_id_to_name: Dict[int, str] = {}
        self.all_class_names: List[str] = []
        self._load_annotations(annotations_file)

    def _cache_path(self) -> Path:
        img_dir_abs = str(Path(self.img_dir).resolve())
        annotations_abs = str(Path(self.annotations_file).resolve())
        stamp = f"{img_dir_abs}|{annotations_abs}"
        digest = hashlib.sha1(stamp.encode("utf-8")).hexdigest()  # nosec B324: non-crypto id is intended
        cache_root = Path(self.cache_dir) if self.cache_dir else Path(tempfile.gettempdir()) / "imagenet_loader_cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        return cache_root / f"{digest}.json"

    def _load_from_cache(self) -> bool:
        if not self.use_cache:
            return False

        cache_path = self._cache_path()
        if not cache_path.exists():
            return False

        try:
            with cache_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return False

        if not isinstance(payload, dict):
            return False

        samples = payload.get("samples")
        class_id_to_name = payload.get("class_id_to_name")
        if not isinstance(samples, list) or not isinstance(class_id_to_name, dict):
            return False

        self.samples = [
            (str(item[0]), int(item[1]), str(item[2]), str(item[3]))
            for item in samples
            if isinstance(item, list) and len(item) == 4
        ]
        self.class_id_to_name = {int(k): str(v) for k, v in class_id_to_name.items()}
        return True

    def _save_cache(self) -> None:
        if not self.use_cache:
            return

        cache_path = self._cache_path()
        payload = {
            "samples": self.samples,
            "class_id_to_name": {str(k): v for k, v in self.class_id_to_name.items()},
        }
        try:
            with cache_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        except Exception:
            # Cache failures should not break training.
            return

    def _load_annotations(self, annotations_file: str) -> None:
        if self._load_from_cache():
            if self.class_id_to_name:
                max_class_id = max(self.class_id_to_name.keys())
                self.all_class_names = ["" for _ in range(max_class_id + 1)]
                for class_id, class_name in self.class_id_to_name.items():
                    self.all_class_names[class_id] = class_name
            return

        with open(annotations_file, "r", encoding="utf-8") as f:
            annotations = json.load(f)  # {folder_name: [class_id, class_name]}

        for folder_name, value in tqdm(
            annotations.items(),
            total=len(annotations),
            desc="Indexing ImageNet folders",
            leave=False,
        ):
            class_id, class_name = value
            class_id = int(class_id)
            class_name = str(class_name).replace("_", " ")
            self.class_id_to_name[class_id] = class_name

            folder_path = os.path.join(self.img_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            with os.scandir(folder_path) as entries:
                for entry in entries:
                    if not entry.is_file():
                        continue
                    img_path = os.path.abspath(entry.path)
                    self.samples.append((img_path, class_id, class_name, folder_name))

        if self.class_id_to_name:
            max_class_id = max(self.class_id_to_name.keys())
            self.all_class_names = ["" for _ in range(max_class_id + 1)]
            for class_id, class_name in self.class_id_to_name.items():
                self.all_class_names[class_id] = class_name

        self._save_cache()

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
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: int = 2,
) -> Tuple[DataLoader, List[str]]:
    dataset = ImageNet(
        img_dir=img_dir,
        annotations_file=annotations_file,
        transform=transform,
        use_cache=use_cache,
        cache_dir=cache_dir,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": True,
    }

    if num_workers > 0:
        if persistent_workers is None:
            loader_kwargs["persistent_workers"] = True
        else:
            loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = max(prefetch_factor, 1)

    loader = DataLoader(dataset, **loader_kwargs)
    return loader, dataset.all_class_names




        
    
        
    
        
        
        