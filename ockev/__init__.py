"""
Ockev: Fast discriminative decision engine for AI agent deliverables.
Inspired by TypeSafe Jev and Occam's razor.
"""

__version__ = "0.1.0"

from .gate import OckevGate
from .model import OckevPointerModel

__all__ = ["OckevGate", "OckevPointerModel", "__version__"]
