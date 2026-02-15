"""
iternet - ERT (electrical resistivity tomography) to geology interpretation.

This package implements:
- Parsers for IE2/IE2D-like formats
- Preprocessing into tensors
- Perceiver-style model: measurements set -> subsurface grid segmentation
- Training utilities with TensorBoard logging
"""

from .config import DataConfig, GridConfig, ModelConfig, TrainConfig  # noqa: F401
