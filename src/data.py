import os
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

# timm provides the ImageNet mean/std used by many pretrained models.
# Importing from timm ensures we stay consistent with any timm backbone.
try:
    from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
except Exception:  # pragma: no cover
    # Fallback values if timm is not installed for some reason.
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def _load_raster(path: str) -> np.ndarray:
    """
    Load a raster file (e.g., GeoTIFF) using rasterio.

    Parameters
    ----------
    path: str
        Path to the raster file.

    Returns
    -------
    np.ndarray
        Array of shape (C, H, W) where C is the number of bands.
        The data are returned as float32.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Raster file not found: {path}")

    with rasterio.open(path) as src:
        # rasterio reads in (band, row, col) order.
        array = src.read(out_dtype="float32")
        # If the raster has a single band, keep the channel dimension.
        if array.ndim == 2:
            array = array[np.newaxis, ...]
    return array


def get_train_transform(
    size: Union[int, Tuple[int, int]] = 224,
    mean: Tuple[float, float, float] = IMAGENET_DEFAULT_MEAN,
    std: Tuple[float, float, float] = IMAGENET_DEFAULT_STD,
) -> Callable:
    """
    Build a transformation pipeline for training.

    The pipeline includes random spatial augmentations, colour jitter,
    conversion to tensor and normalisation.

    Parameters
    ----------
    size : int or tuple
        Target size after cropping/resizing.
    mean, std : tuple
        Normalisation statistics.

    Returns
    -------
    Callable
        A torchvision transform that can be called on a ``PIL.Image`` or ``np.ndarray``.
    """
    if isinstance(size, int):
        size = (size, size)

    return T.Compose(
        [
            T.ToPILImage(),
            T.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def get_val_transform(
    size: Union[int, Tuple[int, int]] = 224,
    mean: Tuple[float, float, float] = IMAGENET_DEFAULT_MEAN,
    std: Tuple[float, float, float] = IMAGENET_DEFAULT_STD,
) -> Callable:
    """
    Build a deterministic transformation pipeline for validation / testing.

    Parameters
    ----------
    size : int or tuple
        Target size after resizing.
    mean, std : tuple
        Normalisation statistics.

    Returns
    -------
    Callable
        A torchvision transform.
    """
    if isinstance(size, int):
        size = (size, size)

    return T.Compose(
        [
            T.ToPILImage(),
            T.Resize(size),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


class SatelliteTileDataset(Dataset):
    """
    PyTorch ``Dataset`` that loads satellite image tiles from disk and applies
    optional augmentations.

    The dataset expects a directory structure like:

    .. code-block:: text

        root/
            tile_001.tif
            tile_002.tif
            ...

    Each file is read with rasterio, converted to ``(C, H, W)`` and then processed
    by the supplied transform.

    Parameters
    ----------
    root_dir : str
        Directory containing the raster tiles.
    file_list : List[str] | None
        List of filenames (relative to ``root_dir``). If ``None``, all ``*.tif``
        files in ``root_dir`` are used.
    transform : Callable | None
        Transform to apply to each image (e.g., ``get_train_transform``).
    """

    def __init__(
        self,
        root_dir: str,
        file_list: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root_dir = os.path.abspath(root_dir)
        if not os.path.isdir(self.root_dir):
            raise NotADirectoryError(f"Root directory does not exist: {self.root_dir}")

        if file_list is None:
            self.file_list = sorted(
                [
                    f
                    for f in os.listdir(self.root_dir)
                    if f.lower().endswith((".tif", ".tiff"))
                ]
            )
        else:
            self.file_list = sorted(file_list)

        if not self.file_list:
            raise RuntimeError(f"No raster files found in {self.root_dir}")

        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Load a tile, apply transforms and return it.

        Returns
        -------
        image : torch.Tensor
            Tensor of shape ``(C, H, W)`` after transformation.
        path : str
            Absolute path to the source raster (useful for debugging).
        """
        filename = self.file_list[idx]
        path = os.path.join(self.root_dir, filename)

        # Load raster data as (C, H, W) float32 numpy array.
        image_np = _load_raster(path)

        # Some satellite products have more than 3 bands; we keep all.
        # Convert to (H, W, C) for torchvision transforms that expect HWC.
        image_np = np.moveaxis(image_np, 0, -1)  # (H, W, C)

        if self.transform is not None:
            image = self.transform(image_np)
        else:
            # Default: convert to tensor and normalise with ImageNet stats.
            image = T.ToTensor()(image_np)
            image = T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)(image)

        return image, path


def create_dataloader(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    train: bool = True,
    image_size: int = 224,
) -> DataLoader:
    """
    Convenience wrapper to create a ``DataLoader`` for the satellite tiles.

    Parameters
    ----------
    root_dir : str
        Directory containing raster tiles.
    batch_size : int
        Number of samples per batch.
    num_workers : int
        Number of subprocesses for data loading.
    shuffle : bool
        Whether to shuffle the dataset (typically ``True`` for training).
    train : bool
        If ``True`` use training augmentations, otherwise validation transforms.
    image_size : int
        Target image size for the model.

    Returns
    -------
    DataLoader
        PyTorch DataLoader ready for iteration.
    """
    transform = get_train_transform(size=image_size) if train else get_val_transform(size=image_size)

    dataset = SatelliteTileDataset(root_dir=root_dir, transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
    )