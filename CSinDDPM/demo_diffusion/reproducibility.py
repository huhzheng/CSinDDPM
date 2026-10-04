"""Seed control and machine-readable run provenance."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Union

import numpy as np
import torch


PathLike = Union[str, os.PathLike]


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch without silently changing algorithms."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False


def current_git_commit(cwd: PathLike) -> str:
    """Return the current commit or an explicit pending marker."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "PENDING_GIT_HISTORY"


def collect_environment() -> Dict[str, Any]:
    """Collect hardware and software details without external services."""

    cuda_available = torch.cuda.is_available()
    gpu_names = []
    gpu_memory_mb = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpu_names.append(props.name)
            gpu_memory_mb.append(round(props.total_memory / (1024 ** 2), 2))
    try:
        import psutil

        ram_mb = round(psutil.virtual_memory().total / (1024 ** 2), 2)
        cpu_count = psutil.cpu_count(logical=True)
    except ImportError:
        ram_mb = None
        cpu_count = os.cpu_count()

    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "pytorch": torch.__version__,
        "cuda_available": cuda_available,
        "pytorch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda_available else None,
        "gpu_names": gpu_names,
        "gpu_memory_mb": gpu_memory_mb,
        "cpu": platform.processor(),
        "logical_cpu_count": cpu_count,
        "ram_mb": ram_mb,
        "numpy": np.__version__,
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, set)):
        return [_json_ready(item) for item in value]
    return value


def write_run_metadata(
    output_dir: PathLike,
    config: Mapping[str, Any],
    run_type: str,
    repository_root: PathLike,
) -> None:
    """Write config, environment, and source revision before a run starts."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    payload = dict(config)
    payload["run_type"] = run_type
    payload["started_at_utc"] = datetime.now(timezone.utc).isoformat()
    (destination / "config.json").write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / "environment.json").write_text(
        json.dumps(collect_environment(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    commit = current_git_commit(repository_root)
    (destination / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")


def build_model_summary(model, requested_attention=None) -> Dict[str, Any]:
    """Return architecture facts that cannot be inferred safely from CLI names."""

    condition_encoder = getattr(model, "cond_embedding2", None)
    resblocks = [
        module
        for module in model.modules()
        if module.__class__.__name__ == "ResBlock"
    ]
    conditional_resblocks = [
        module
        for module in resblocks
        if getattr(module, "cond_layers", None) is not None
    ]
    condition_encoder_parameter_count = (
        sum(parameter.numel() for parameter in condition_encoder.parameters())
        if condition_encoder is not None
        else 0
    )
    conditional_resblock_parameter_count = sum(
        parameter.numel()
        for module in conditional_resblocks
        for parameter in module.cond_layers.parameters()
    )
    use_condition = bool(getattr(model, "use_condition", False))
    summary = {
        "model_class": model.__class__.__name__,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "attention_block_count": sum(
            module.__class__.__name__ == "AttentionBlock" for module in model.modules()
        ),
        "requested_attention_resolutions": requested_attention,
        "use_condition": use_condition,
        "condition_encoder_parameter_count": condition_encoder_parameter_count,
        "resblock_count": len(resblocks),
        "conditional_resblock_count": len(conditional_resblocks),
        "conditional_resblock_parameter_count": (
            conditional_resblock_parameter_count
        ),
        "conditional_parameter_count": (
            condition_encoder_parameter_count
            + conditional_resblock_parameter_count
        ),
        "condition_fusion": (
            "additive_residual_fusion" if use_condition else "disabled"
        ),
        "phase_convention": {"pore": 1.0, "grain": -1.0},
    }
    return summary


def write_model_summary(output_dir: PathLike, model, requested_attention=None) -> Dict[str, Any]:
    """Write a machine-readable architecture summary and return it."""

    summary = build_model_summary(model, requested_attention=requested_attention)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "model_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
