from __future__ import annotations

import logging
import subprocess
from functools import lru_cache
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


@lru_cache(maxsize=1)
def cuda_available() -> bool:
    """True only when a CUDA session can actually be created AND bound.

    ort.get_available_providers() lists the providers COMPILED INTO the build,
    not the ones that can load. It returns CUDAExecutionProvider even when
    libonnxruntime_providers_cuda.so fails to open - a cuDNN major-version
    mismatch does exactly that - and every session then falls back to CPU
    silently. On 2026-09-02 /health reported gpu_available: true while nothing
    ran on the GPU. The only honest test is to build a session and ask what it
    actually bound.

    Cached: the probe is far too costly to run on every /health request, and the
    answer cannot change within a process.
    """
    try:
        import onnxruntime as ort
        from onnx import TensorProto, helper
    except ImportError:
        return False

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return False

    try:
        graph = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["y"])], "cuda_probe",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 9        # ort 1.18 rejects onnx 1.22's newer default
        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # the fallback warning is ours to interpret
        sess = ort.InferenceSession(
            model.SerializeToString(), opts, providers=["CUDAExecutionProvider"]
        )
        return "CUDAExecutionProvider" in sess.get_providers()
    except Exception as exc:
        logger.warning("CUDA probe failed; treating GPU as unavailable: %s", exc)
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
