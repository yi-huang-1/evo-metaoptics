from __future__ import annotations

import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_NVIDIA_SMI_QUERY_FIELDS = [
    "index",
    "name",
    "uuid",
    "memory.total",
    "memory.used",
    "memory.free",
    "utilization.gpu",
    "utilization.memory",
    "temperature.gpu",
    "power.draw",
    "driver_version",
    "cuda_version",
]


def validate_data_files(*, train_data: str, val_data: str, test_data: str | None = None) -> dict[str, str]:
    train_path = Path(train_data)
    val_path = Path(val_data)
    if not train_path.is_file():
        raise FileNotFoundError(f"Train data not found: {train_data}")
    if not val_path.is_file():
        raise FileNotFoundError(f"Validation data not found: {val_data}")
    result = {"train_data": str(train_path), "val_data": str(val_path)}
    if test_data is not None:
        test_path = Path(test_data)
        if not test_path.is_file():
            raise FileNotFoundError(f"Held-out test data not found: {test_data}")
        result["test_data"] = str(test_path)
    return result


def validate_model_configured(*, model: str) -> str:
    normalized = str(model).strip()
    if not normalized:
        raise ValueError("Model not configured. Set experiment.model in YAML or environment.")
    return normalized


def capture_gpu_snapshot(*, run_name: str, output_path: Path) -> dict[str, Any]:
    query = ",".join(_NVIDIA_SMI_QUERY_FIELDS)
    snapshot: dict[str, Any] = {
        "schema_version": 1,
        "snapshot_type": "gpu_status",
        "run_name": run_name,
        "captured_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "hostname": socket.gethostname(),
        "nvidia_smi_available": False,
        "probe_success": False,
        "probe_error": None,
        "gpu_count": 0,
        "devices": [],
        "raw": {
            "nvidia_smi_query": query,
            "nvidia_smi_stdout": "",
            "nvidia_smi_stderr": "",
        },
    }

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        snapshot["nvidia_smi_available"] = True
        snapshot["raw"]["nvidia_smi_stdout"] = result.stdout
        snapshot["raw"]["nvidia_smi_stderr"] = result.stderr
        if result.returncode != 0:
            snapshot["probe_error"] = f"nvidia-smi exited with code {result.returncode}: {result.stderr.strip()}"
        else:
            devices = _parse_nvidia_smi_output(result.stdout)
            snapshot["probe_success"] = True
            snapshot["gpu_count"] = len(devices)
            snapshot["devices"] = devices
    except FileNotFoundError as exc:
        snapshot["probe_error"] = str(exc)
    except Exception as exc:
        snapshot["nvidia_smi_available"] = True
        snapshot["probe_error"] = str(exc)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return snapshot


def _parse_nvidia_smi_output(stdout: str) -> list[dict[str, Any]]:
    devices: list[dict[str, Any]] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        padded = parts + [""] * max(0, len(_NVIDIA_SMI_QUERY_FIELDS) - len(parts))
        devices.append(
            {
                "index": _to_int(padded[0]),
                "name": padded[1],
                "uuid": padded[2],
                "memory_total_mb": _to_int(padded[3]),
                "memory_used_mb": _to_int(padded[4]),
                "memory_free_mb": _to_int(padded[5]),
                "utilization_gpu_pct": _to_int(padded[6]),
                "utilization_memory_pct": _to_int(padded[7]),
                "temperature_c": _to_int(padded[8]),
                "power_w": _to_float(padded[9]),
                "driver_version": padded[10],
                "cuda_version": padded[11],
            }
        )
    return devices


def _to_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "capture_gpu_snapshot",
    "validate_data_files",
    "validate_model_configured",
]
