"""Segmentation models package."""

from .base import BaseSegmentor
from .kmax_wrapper import KMaxSegmentor
from .fcclip_wrapper import FCCLIPSegmentor

__all__ = ['BaseSegmentor', 'KMaxSegmentor', 'FCCLIPSegmentor']
