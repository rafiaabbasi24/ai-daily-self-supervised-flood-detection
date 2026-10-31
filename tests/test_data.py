import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin
from torch.utils.data import DataLoader

# The project is expected to expose a dataset class and a transform factory.
# Adjust the import path according to the actual package structure.
from flood_detection.dataset import FloodDataset, get_transforms


@pytest.fixture(scope="module")
def dummy_satellite_data():
    """
    Create a temporary directory with a few small GeoTIFF files and a CSV
    containing fake flood labels. The fixture yields the directory path.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create a simple 3‑band raster (e.g., RGB) of size 64x64.
        def _write_tif(name: str, array: np.ndarray):
            transform = from_origin(0, 0, 1, 1)  # arbitrary georeferencing
            profile = {
                "driver": "GTiff",
                "height": array.shape[1],
                "width": array.shape[2],
                "count": array.shape[0],
                "dtype": array.dtype,
                "crs": "EPSG:4326",
                "transform": transform,
            }
            with rasterio.open(root / name, "w", **profile) as dst:
                for i in range(array.shape[0]):
                    dst.write(array[i], i + 1)

        # Generate three dummy images.
        rng = np.random.default_rng(42)
        for i in range(3):
            img = rng.integers(0, 255, size=(3, 64, 64), dtype=np.uint8)
            _write_tif(f"image_{i}.tif", img)

        # Create a CSV with image filenames and binary flood labels.
        csv_path = root / "labels.csv"
        csv_path.write_text(
            "\n".join(
                [
                    "filename,label",
                    "image_0.tif,0",
                    "image_1.tif,1",
                    "image_2.tif,0",
                ]
            )
        )

        yield root


def test_dataset_initialisation(dummy_satellite_data):
    """
    Verify that the FloodDataset can be instantiated and reports the correct length.
    """
    dataset = FloodDataset(
        root_dir=str(dummy_satellite_data),
        csv_file="labels.csv",
        transform=get_transforms(train=False),
    )
    assert len(dataset) == 3, "Dataset length should match number of images"


def test_dataset_item_structure(dummy_satellite_data):
    """
    Check that a single item from the dataset returns a tensor image and an integer label.
    """
    dataset = FloodDataset(
        root_dir=str(dummy_satellite_data),
        csv_file="labels.csv",
        transform=get_transforms(train=False),
    )
    img, label = dataset[0]

    # Image must be a torch.Tensor with 3 channels.
    assert isinstance(img, torch.Tensor), "Image should be a torch.Tensor"
    assert img.ndim == 3, "Image tensor should have 3 dimensions (C, H, W)"
    assert img.shape[0] == 3, "Image should have 3 channels (RGB)"

    # Label must be an int or torch scalar convertible to int.
    assert isinstance(label, (int, np.integer, torch.Tensor)), "Label type is incorrect"
    if isinstance(label, torch.Tensor):
        label = label.item()
    assert label in (0, 1), "Label should be binary (0 or 1)"


def test_transform_output():
    """
    Ensure that the transformation pipeline returns tensors of the expected dtype and range.
    """
    transform = get_transforms(train=False)

    # Create a dummy 3‑channel uint8 image.
    dummy_image = np.random.randint(0, 256, size=(3, 128, 128), dtype=np.uint8)

    # The transform is expected to accept a numpy array or PIL image.
    # Convert to a torch tensor via the transform.
    transformed = transform(dummy_image)

    assert isinstance(transformed, torch.Tensor), "Transform output must be a torch.Tensor"
    assert transformed.dtype == torch.float32, "Transformed tensor should be float32"
    # Typical self‑supervised pipelines normalise to [0, 1] or have zero‑mean unit‑var.
    assert transformed.min() >= 0.0 and transformed.max() <= 1.0, (
        "Transformed tensor values should be in the range [0, 1]"
    )


def test_dataloader_batching(dummy_satellite_data):
    """
    Validate that a DataLoader can batch samples from FloodDataset without errors.
    """
    dataset = FloodDataset(
        root_dir=str(dummy_satellite_data),
        csv_file="labels.csv",
        transform=get_transforms(train=False),
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    batch = next(iter(loader))
    images, labels = batch

    assert isinstance(images, torch.Tensor), "Batch images should be a torch.Tensor"
    assert images.shape[0] == 2, "Batch size should be 2"
    assert images.shape[1] == 3, "Each image should have 3 channels"
    assert isinstance(labels, torch.Tensor), "Batch labels should be a torch.Tensor"
    assert labels.shape[0] == 2, "Batch labels should match batch size"
    assert labels.dtype == torch.long or labels.dtype == torch.int64, (
        "Labels tensor should be of integer type"
    )