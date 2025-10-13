"""
Certificate Text Overlay Package

A ComfyUI-based utility for adding text overlays to certificate images.
"""

from .certificate_text_adder import (
    ComfyUICertificateTextAdder,
    add_name_to_certificate
)

__version__ = "1.0.0"
__author__ = "ComfyUI Certificate Text Overlay"
__description__ = "Add text overlays to certificate images using ComfyUI and PIL"

__all__ = [
    "ComfyUICertificateTextAdder",
    "add_name_to_certificate"
]
