"""
CUDA Haar Wavelet Transform Library.

Unified library for 2D Haar wavelet transforms with multiple cascade levels.
"""

from .haar_cuda import (
    # Wrapper classes
    HaarCUDA,
    HaarDoubleCUDA,
    HaarDoubleTransform,
    HaarQuadCUDA,
    HaarQuadTransform,
    HaarQuintCUDA,
    HaarQuintTransform,
    # Autograd functions (for advanced use)
    HaarTransform,
    HaarTripleCUDA,
    HaarTripleTransform,
    InverseHaarTransform,
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
)

__all__ = [
    "HaarCUDA",
    "HaarDoubleCUDA",
    "HaarDoubleTransform",
    "HaarQuadCUDA",
    "HaarQuadTransform",
    "HaarQuintCUDA",
    "HaarQuintTransform",
    "HaarTransform",
    "HaarTripleCUDA",
    "HaarTripleTransform",
    "InverseHaarTransform",
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
]
