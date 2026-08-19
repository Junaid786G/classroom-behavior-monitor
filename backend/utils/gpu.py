from __future__ import annotations

import logging
import subprocess
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)

# ── ONNX provider helpers ─────────────────────────────────────────────────────

ProviderSpec = Union[str, tuple]  # 'CPUExecutionProvider' or ('CUDA...', {opts})


def get_onnx_providers(ctx_id: int = 0) -> List[ProviderSpec]:
    """
    Return an ordered provider list for onnxruntime.
    CUDAExecutionProvider is preferred when available and ctx_id >= 0.
    """
    try:
        import onnxruntime as ort

        available = ort.get_available_providers()
        logger.debug(f"Available ONNX providers: {available}")

        if "CUDAExecutionProvider" in available and ctx_id >= 0:
            cuda_opts: Dict[str, Any] = {
                "device_id": ctx_id,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            }
            logger.info(f"ONNX: CUDAExecutionProvider selected (device {ctx_id})")
            return [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]

    except ImportError:
        logger.warning("onnxruntime not importable – falling back to CPU")

    logger.info("ONNX: CPUExecutionProvider selected")
    return ["CPUExecutionProvider"]


def cuda_available() -> bool:
    try:
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except ImportError:
        return False


# ── Device info ───────────────────────────────────────────────────────────────

def get_device_info() -> Dict[str, Any]:
    """Return a dict summarising GPU/CPU availability – used in /health."""
    info: Dict[str, Any] = {
        "cuda_available": False,
        "onnx_providers": [],
        "gpu_devices": [],
    }

    try:
        import onnxruntime as ort
        info["onnx_providers"] = ort.get_available_providers()
        info["cuda_available"] = "CUDAExecutionProvider" in info["onnx_providers"]
        info["onnxruntime_version"] = ort.__version__
    except ImportError:
        pass

    if info["cuda_available"]:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 5:
                        info["gpu_devices"].append(
                            {
                                "index": int(parts[0]),
                                "name": parts[1],
                                "memory_total_mb": int(parts[2]),
                                "memory_free_mb": int(parts[3]),
                                "utilization_pct": int(parts[4]),
                            }
                        )
        except Exception as exc:
            logger.debug(f"nvidia-smi query failed: {exc}")

    return info


# ── InsightFace ctx_id ────────────────────────────────────────────────────────

def insightface_ctx_id(setting_ctx_id: int) -> int:
    """
    Return the ctx_id to pass to insightface model.prepare().
    -1 forces CPU; 0+ selects that GPU device.
    Trusts the config directly — ONNX runtime falls back to CPU if CUDA is
    not available, so we do not gate here on cuda_available().
    """
    return setting_ctx_id
