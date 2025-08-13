# pytorch_io.py
from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, Optional, Tuple, Callable

import torch

MODEL_DIR = "saved_models"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _atomic_save(obj: Any, path: str) -> None:
    """Write a file atomically to avoid partial/corrupted checkpoints."""
    _ensure_dir(os.path.dirname(path) or ".")
    fd, tmppath = tempfile.mkstemp(dir=os.path.dirname(path) or ".", prefix=".tmp_", suffix=".pt")
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(obj, f)
        os.replace(tmppath, path)
    finally:
        if os.path.exists(tmppath):
            try:
                os.remove(tmppath)
            except OSError:
                pass


def save_model(
    model: torch.nn.Module,
    model_name: str = "model.pt",
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
    state_dict_only: bool = False,
) -> str:
    """
    Save a PyTorch model checkpoint.

    Args:
        model: Your torch.nn.Module.
        model_name: Filename under MODEL_DIR.
        optimizer: (optional) Save optimizer state.
        scheduler: (optional) Save LR scheduler state.
        epoch: (optional) Last completed epoch.
        metrics: (optional) Dict of metrics to store.
        extra: (optional) Any additional metadata (e.g., config).
        state_dict_only: If True, save only model.state_dict() for maximum portability.

    Returns:
        The saved path.
    """
    path = os.path.join(MODEL_DIR, model_name)
    _ensure_dir(MODEL_DIR)

    if state_dict_only and not any([optimizer, scheduler, epoch, metrics, extra]):
        # Simple/portable format (compatible with your original snippet)
        _atomic_save(model.state_dict(), path)
    else:
        # Rich checkpoint
        ckpt: Dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics or {},
            "extra": extra or {},
            "torch_version": torch.__version__,
        }
        if optimizer is not None:
            ckpt["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            ckpt["scheduler_state_dict"] = scheduler.state_dict()

        _atomic_save(ckpt, path)

    print(f"Model saved at {path}")
    return path


def load_model(
    model_builder: Callable[[], torch.nn.Module],
    model_name: str = "model.pt",
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: Optional[torch.device | str] = None,
    strict: bool = True,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load a model (and optionally optimizer/scheduler) from disk.

    Args:
        model_builder: Zero-arg callable that returns a freshly initialized model
                       with the same architecture used at save time.
        model_name: Filename under MODEL_DIR.
        optimizer: (optional) Optimizer to load state into.
        scheduler: (optional) Scheduler to load state into.
        map_location: Device mapping for loading (e.g., "cpu" or torch.device("cuda")).
        strict: Passed to load_state_dict to enforce exact key matching.

    Returns:
        (model, info) where info contains epoch/metrics/extra if available.
    """
    path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # If map_location not provided, choose a sensible default
    if map_location is None:
        map_location = "cuda" if torch.cuda.is_available() else "cpu"

    raw = torch.load(path, map_location=map_location)

    model = model_builder()
    info: Dict[str, Any] = {}

    if isinstance(raw, dict) and "model_state_dict" in raw:
        # Rich checkpoint path
        missing, unexpected = model.load_state_dict(raw["model_state_dict"], strict=strict)
        if optimizer is not None and "optimizer_state_dict" in raw:
            try:
                optimizer.load_state_dict(raw["optimizer_state_dict"])
            except Exception:
                pass  # Don't fail the whole load if optimizer state mismatches
        if scheduler is not None and "scheduler_state_dict" in raw:
            try:
                scheduler.load_state_dict(raw["scheduler_state_dict"])
            except Exception:
                pass

        info = {
            "epoch": raw.get("epoch"),
            "metrics": raw.get("metrics", {}),
            "extra": raw.get("extra", {}),
            "missing_keys": missing,
            "unexpected_keys": unexpected,
        }
    else:
        # state_dict-only path
        missing, unexpected = model.load_state_dict(raw, strict=strict)
        info = {"missing_keys": missing, "unexpected_keys": unexpected}

    model.eval()
    print(f"Model loaded from {path} (map_location={map_location})")
    return model, info
def list_saved_models() -> Dict[str, str]:
    """ List all saved models in MODEL_DIR. """
    _ensure_dir(MODEL_DIR)