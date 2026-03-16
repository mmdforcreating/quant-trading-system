"""
硬件自适应设备选择模块

优先级: Apple MPS (Metal Performance Shaders) > CPU
Mac 上的 PyTorch 从 1.12 起支持 MPS 后端，可利用 Apple Silicon GPU 加速矩阵运算。
mps 后端对部分算子（如某些 RNN 变体）仍有兼容性问题，故提供 fallback 开关。
"""
from __future__ import annotations

import torch


def get_device(force_cpu: bool = False) -> torch.device:
    """
    自动检测并返回最优计算设备。

    检测顺序：
    1. 若 force_cpu=True，直接返回 CPU（调试用）
    2. 检测 MPS 是否可用（Apple Silicon Mac）
    3. 回退到 CPU

    Returns
    -------
    torch.device
    """
    if force_cpu:
        return torch.device("cpu")

    # Apple MPS: M1/M2/M3/M4 系列芯片的 GPU 加速
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            # 验证 MPS 实际可用（某些 macOS 版本虽报告可用但实际不工作）
            torch.zeros(1, device="mps")
            return torch.device("mps")
        except Exception:
            pass

    return torch.device("cpu")


def get_device_info() -> str:
    """返回当前设备的可读描述。"""
    device = get_device()
    if device.type == "mps":
        return f"Apple MPS (Metal Performance Shaders) - {device}"
    return f"CPU - {device}"
