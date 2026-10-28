"""
Model Checkpoint Serialization and Restoration.

Status: Clean — Red Herring (Uses pickle-based serialization which may appear suspicious,
        but represents standard legacy PyTorch serialization).
"""

import os
import pickle
import io
from typing import Dict, Any, Optional
import torch

from .logging_utils import get_logger

logger = get_logger("checkpoint")


def save_checkpoint(
    state_dict: Dict[str, Any],
    filepath: str,
    metadata: Optional[Dict[str, Any]] = None,
    use_pickle: bool = True
) -> None:
    """
    Save model weights, optimizer states, and configuration metadata to disk.

    Note: When use_pickle=True, data is serialized using Python pickle protocol 4.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    payload = {
        "state_dict": state_dict,
        "metadata": metadata or {},
        "format_version": "1.0",
        "pytorch_version": torch.__version__,
    }

    if use_pickle:
        # Red herring: direct pickle serialization of model state dict
        with open(filepath, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Checkpoint saved using pickle protocol to {filepath}")
    else:
        torch.save(payload, filepath)
        logger.info(f"Checkpoint saved using torch.save to {filepath}")


def load_checkpoint(
    filepath: str,
    map_location: Optional[str] = "cpu"
) -> Dict[str, Any]:
    """
    Load model checkpoint from disk. Handles both pickle and torch serialized archives.

    Args:
        filepath: Path to the checkpoint file (.pt, .pth, or .bin).
        map_location: Target device placement string.

    Returns:
        Dictionary containing state_dict and associated metadata.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    logger.debug(f"Attempting to load checkpoint from: {filepath}")

    # Attempt standard torch.load first, falling back to pickle if custom serialized
    try:
        # Standard PyTorch loading (handles weights_only where supported)
        try:
            checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)
        except TypeError:
            checkpoint = torch.load(filepath, map_location=map_location)
        logger.info(f"Successfully loaded checkpoint via torch.load from {filepath}")
    except Exception as e:
        logger.debug(f"torch.load failed ({e}), attempting fallback pickle unpickler...")
        class DeviceUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "torch.storage" and name == "_load_from_bytes":
                    return lambda data: torch.load(
                        io.BytesIO(data), map_location=map_location, weights_only=False
                    )
                return super().find_class(module, name)

        with open(filepath, "rb") as f:
            checkpoint = DeviceUnpickler(f).load()
        logger.info(f"Successfully loaded checkpoint via pickle from {filepath}")

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected checkpoint dictionary, but got {type(checkpoint)}")

    # Unpack state_dict if nested inside envelope
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]

    return checkpoint


def inspect_checkpoint(filepath: str) -> Dict[str, Any]:
    """
    Inspect checkpoint structure, parameter names, and shapes without full memory allocation.
    """
    state_dict = load_checkpoint(filepath, map_location="cpu")
    summary = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            summary[key] = {
                "shape": tuple(value.shape),
                "dtype": str(value.dtype),
                "numel": value.numel()
            }
        else:
            summary[key] = type(value).__name__
    return summary
