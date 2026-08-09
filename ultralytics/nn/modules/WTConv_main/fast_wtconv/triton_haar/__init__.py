"""
Triton Haar Wavelet Transform Module.

Provides fused Haar → Conv → Scale kernels for Hybrid WTConv.
"""

from .triton_haar import (
    compute_scaled_weight,
    fused_haar_conv_scale,
)

__all__ = [
    "compute_scaled_weight",
    "fused_haar_conv_scale",
]
