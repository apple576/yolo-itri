"""
Metal Haar Wavelet Transform Library.

Unified library for 2D Haar wavelet transforms on Apple Metal.
Requires macOS with Metal-capable GPU and PyTorch MPS support.
"""

from .haar_metal import (
    HaarDoubleMetal,
    HaarDoubleTransform,
    # Wrapper classes
    HaarMetal,
    HaarQuadMetal,
    HaarQuadTransform,
    HaarQuintMetal,
    HaarQuintTransform,
    # Autograd functions (for advanced use)
    HaarTransform,
    HaarTripleMetal,
    HaarTripleTransform,
    InverseHaarDoubleTransform,
    InverseHaarQuadTransform,
    InverseHaarQuintTransform,
    InverseHaarTransform,
    InverseHaarTripleTransform,
    ScaledDepthwiseConvFunction,
    # Functional API
    haar2d,
    haar2d_double,
    haar2d_quad,
    haar2d_quint,
    haar2d_triple,
    ihaar2d,
    ihaar2d_double,
    ihaar2d_quad,
    ihaar2d_quint,
    ihaar2d_triple,
    # Scaled depthwise conv
    scaled_depthwise_conv,
)

__all__ = [
    "HaarDoubleMetal",
    "HaarDoubleTransform",
    "HaarMetal",
    "HaarQuadMetal",
    "HaarQuadTransform",
    "HaarQuintMetal",
    "HaarQuintTransform",
    "HaarTransform",
    "HaarTripleMetal",
    "HaarTripleTransform",
    "InverseHaarDoubleTransform",
    "InverseHaarQuadTransform",
    "InverseHaarQuintTransform",
    "InverseHaarTransform",
    "InverseHaarTripleTransform",
    "ScaledDepthwiseConvFunction",
    "haar2d",
    "haar2d_double",
    "haar2d_quad",
    "haar2d_quint",
    "haar2d_triple",
    "ihaar2d",
    "ihaar2d_double",
    "ihaar2d_quad",
    "ihaar2d_quint",
    "ihaar2d_triple",
    "scaled_depthwise_conv",
]
