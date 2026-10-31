"""tests/test_model.py
Unit tests for the flood detection model's forward pass.

These tests verify that the model can be instantiated, processes inputs of the
expected shape, and returns outputs with the correct dimensions. They also
exercise common usage patterns such as inference under ``torch.no_grad`` and
running on both CPU and (if available) GPU.

The tests assume the model implementation lives in
``src/models/flood_detection.py`` and provides a class named
``FloodDetectionModel``. Adjust the import path if the project structure differs.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

# Import the model under test. Adjust the import if the module location is different.
try:
    from src.models.flood_detection import FloodDetectionModel
except ImportError as exc:
    raise ImportError(
        "Could not import FloodDetectionModel. Ensure the project structure "
        "contains 'src/models/flood_detection.py' with a class named "
        "'FloodDetectionModel'."
    ) from exc


@pytest.fixture(scope="module")
def model() -> FloodDetectionModel:
    """
    Create a fresh instance of the model for the test session.

    Returns
    -------
    FloodDetectionModel
        An untrained model ready for a forward pass.
    """
    # The model may accept configuration arguments; use defaults.
    return FloodDetectionModel()


@pytest.fixture(params=[torch.device("cpu"), torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")])
def device(request) -> torch.device:
    """
    Parameterized fixture that yields a device (CPU or GPU) for testing.

    Returns
    -------
    torch.device
        The device on which the model and tensors will be placed.
    """
    return request.param


@pytest.fixture
def dummy_input(device: torch.device) -> torch.Tensor:
    """
    Create a dummy batch of satellite images.

    The model is expected to accept a 4‑D tensor with shape
    (batch_size, channels, height, width).  Satellite imagery often contains
    three RGB channels; adjust ``channels`` if the model expects a different
    number.

    Parameters
    ----------
    device : torch.device
        Device on which the tensor will be allocated.

    Returns
    -------
    torch.Tensor
        Random float tensor simulating a batch of images.
    """
    batch_size = 2
    channels = 3          # RGB; change if the model expects more channels
    height = 256
    width = 256
    torch.manual_seed(0)  # deterministic values for reproducibility
    return torch.randn(batch_size, channels, height, width, device=device)


def test_model_is_nn_module(model: FloodDetectionModel) -> None:
    """The model should be a subclass of ``torch.nn.Module``."""
    assert isinstance(model, nn.Module), "FloodDetectionModel must inherit from torch.nn.Module"


def test_forward_output_shape(model: FloodDetectionModel, dummy_input: torch.Tensor) -> None:
    """
    Verify that a forward pass returns a tensor of the expected shape.

    For binary flood segmentation the typical output shape is
    (batch_size, 1, height, width) – a per‑pixel logit map.
    Adjust the expected shape if the model implements a different task.
    """
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    assert isinstance(output, torch.Tensor), "Model output must be a torch.Tensor"
    expected_shape = (dummy_input.shape[0], 1, dummy_input.shape[2], dummy_input.shape[3])
    assert output.shape == expected_shape, (
        f"Unexpected output shape: got {output.shape}, expected {expected_shape}"
    )
    # The output should be of floating point type (logits)
    assert output.dtype.is_floating_point, "Model output should be a floating point tensor"


def test_forward_no_grad(model: FloodDetectionModel, dummy_input: torch.Tensor) -> None:
    """
    Ensure the model runs without tracking gradients during inference.
    """
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    # After the no_grad context, gradients should not be attached.
    assert not output.requires_grad, "Output should not require gradients under torch.no_grad()"


def test_device_compatibility(model: FloodDetectionModel, dummy_input: torch.Tensor, device: torch.device) -> None:
    """
    Confirm that the model and inputs can be moved to the specified device and
    still produce a valid forward pass.
    """
    # Move model parameters to the target device
    model = model.to(device)
    # Ensure the dummy input is already on the correct device (fixture guarantees this)
    assert dummy_input.device == device, "Dummy input is not on the expected device"

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    # Output should reside on the same device as the input
    assert output.device == device, "Model output is not on the expected device"
    # Shape sanity check (reuse logic from earlier test)
    expected_shape = (dummy_input.shape[0], 1, dummy_input.shape[2], dummy_input.shape[3])
    assert output.shape == expected_shape, "Output shape mismatch on device test"


def test_multiple_forward_calls_consistency(model: FloodDetectionModel, dummy_input: torch.Tensor) -> None:
    """
    Run the forward pass twice and ensure the outputs are consistent in shape
    and datatype. This catches accidental stateful behavior.
    """
    model.eval()
    with torch.no_grad():
        out1 = model(dummy_input)
        out2 = model(dummy_input)

    assert out1.shape == out2.shape, "Inconsistent output shapes across forward calls"
    assert out1.dtype == out2.dtype, "Inconsistent output dtypes across forward calls"
    # The values may differ due to dropout (if not disabled), but shape/dtype must be stable.
    assert not torch.isnan(out1).any(), "Output contains NaNs on first forward pass"
    assert not torch.isnan(out2).any(), "Output contains NaNs on second forward pass"
