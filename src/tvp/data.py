# src/tvp/data.py
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from typing import Callable, Optional, Dict, List, Tuple, Any

__all__ = [
    "DataModule",
    "VALID_DATASETS"
]

VALID_DATASETS: List[str] = [
    "CIFAR10", "CIFAR100", "EMNIST", "EuroSAT", "FashionMNIST",
    "Flowers102", "Food101", "MNIST", "OxfordIIITPet", "SVHN"
]

# A set of known grayscale datasets for automatic channel conversion.
GRAY_DATASETS: set[str] = {"MNIST", "FashionMNIST", "EMNIST"}


def _wrap_for_rgb(preprocess: Optional[Callable]) -> Callable:
    if preprocess is None:
        # Fallback to a basic ToTensor if no preprocess is provided.
        return T.Compose([T.Grayscale(num_output_channels=3), T.ToTensor()])
    
    # Check if preprocess is a Compose object to safely insert the transform.
    if isinstance(preprocess, T.Compose):
        # Create a new list of transforms to avoid modifying the original object.
        new_transforms = [T.Grayscale(num_output_channels=3)] + preprocess.transforms
        return T.Compose(new_transforms)
    else:
        # If it's not a Compose object, wrap it in a new one.
        return T.Compose([T.Grayscale(num_output_channels=3), preprocess])


class DataModule:
    def __init__(self, config: Dict, preprocess: Optional[Callable] = None):
        dataset_cfg = config["data"]
        dataset_name_in = dataset_cfg["name"]
        data_dir = dataset_cfg.get("path", "data")
        batch_size = dataset_cfg.get("batch_size", 128)
        num_workers = dataset_cfg.get("num_workers", 4)
        seed = int(dataset_cfg.get("seed", 42))
        use_pin = bool(dataset_cfg.get("pin_memory", torch.cuda.is_available()))
        
        try:
            use_persist = bool(dataset_cfg.get("persistent_workers", num_workers > 0))
            loader_kwargs = {"persistent_workers": use_persist}
        except TypeError:
            loader_kwargs = {}

        # 1) Official name resolve
        dataset_name = self._find_official_name(dataset_name_in)

        # 2) Select dataset class
        try:
            dataset_class = getattr(torchvision.datasets, dataset_name)
        except AttributeError:
            raise ImportError(f"Could not import '{dataset_name}' from torchvision.datasets.")

        # 3) Preprocess handling (grayscale datasets → RGB 3ch)
        eff_preprocess = _wrap_for_rgb(preprocess) if dataset_name in GRAY_DATASETS else preprocess

        # 4) Instantiate train/test datasets
        if dataset_name == "EuroSAT":
            full_ds = dataset_class(root=data_dir, download=True, transform=eff_preprocess)
            g = torch.Generator().manual_seed(seed)
            train_size = int(0.8 * len(full_ds))
            val_size = len(full_ds) - train_size
            self.train_dataset, self.test_dataset = random_split(full_ds, [train_size, val_size], generator=g)
        else:
            train_args, test_args = self._get_dataset_args(dataset_name, data_dir, eff_preprocess, dataset_cfg)
            train_ds = dataset_class(**train_args)
            test_ds = dataset_class(**test_args)

            if dataset_name == "SVHN" and dataset_cfg.get("use_extra", False):
                extra_args = train_args.copy()
                extra_args["split"] = "extra"
                extra_ds = dataset_class(**extra_args)
                train_ds = ConcatDataset([train_ds, extra_ds])

            self.train_dataset, self.test_dataset = train_ds, test_ds

        # 5) DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=use_pin, **loader_kwargs
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=use_pin, **loader_kwargs
        )

        # 6) Class names (robust)
        self.classnames = self._get_classnames(self.train_dataset)
        if self.classnames is None:
            print(f"Warning: Could not automatically determine class names for {dataset_name}.")


    def _find_official_name(self, name: str) -> str:
        name_lower = name.lower()
        for valid_name in VALID_DATASETS:
            if valid_name.lower() == name_lower:
                return valid_name
        raise ValueError(f"Unsupported dataset: '{name}'. Supported: {VALID_DATASETS}")


    def _get_dataset_args(self, name: str, data_dir: str, preprocess: Callable, cfg: Dict) -> Tuple[Dict, Dict]:
        train_args: Dict[str, Any] = {"root": data_dir, "download": True, "transform": preprocess}
        test_args: Dict[str, Any] = train_args.copy()

        if name in ["SVHN", "OxfordIIITPet", "Flowers102", "Food101"]:
            train_args["split"] = "train"
            test_args["split"] = "test"
            if name == "OxfordIIITPet":
                train_args["split"] = "trainval"
        elif name == "EMNIST":
            split_type = cfg.get("emnist_split", "byclass")
            train_args.update({"split": split_type, "train": True})
            test_args.update({"split": split_type, "train": False})
        else:
            train_args["train"] = True
            test_args["train"] = False

        return train_args, test_args


    def _base_dataset(self, ds: Dataset) -> Dataset:
        current_ds = ds
        while isinstance(current_ds, (torch.utils.data.Subset, ConcatDataset)):
            if isinstance(current_ds, torch.utils.data.Subset):
                current_ds = current_ds.dataset
            elif isinstance(current_ds, ConcatDataset):
                # We can only reliably get metadata if all datasets are the same type.
                # Taking the first one is a reasonable heuristic.
                current_ds = current_ds.datasets[0]
        return current_ds


    def _get_classnames(self, ds: Dataset) -> Optional[List[str]]:
        base = self._base_dataset(ds)

        # 1) Most datasets expose .classes
        if hasattr(base, "classes") and isinstance(base.classes, list):
            return base.classes

        # 2) Fallback to class_to_idx
        if hasattr(base, "class_to_idx") and isinstance(base.class_to_idx, dict):
            # Sort by index to ensure consistent class order.
            idx_to_class = {idx: cls for cls, idx in base.class_to_idx.items()}
            return [idx_to_class[i] for i in sorted(idx_to_class)]

        # 3) Dataset-specific hardcoded fallbacks
        if isinstance(base, torchvision.datasets.SVHN):
            return [str(i) for i in range(10)]
        if isinstance(base, torchvision.datasets.MNIST):
            return [str(i) for i in range(10)]
        
        # 4) Unknown
        return None
    