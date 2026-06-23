"""Temporal motion modules ported from Video-Depth-Anything.

These are the cross-frame attention blocks (``TemporalModule``) that turn the
spatial Depth-Anything-V2 DPT head into a temporally-consistent video head.
The two files in this package (``motion_module.py``, ``attention.py``) are
vendored verbatim from Video-Depth-Anything (Apache-2.0), which in turn derives
``motion_module.py`` from AnimateDiff. Vendoring keeps VJEDAI self-contained and
lets the VDA pretrained ``head.motion_modules.*`` weights load key-for-key.

The only public symbol used by :mod:`vjedai` is
:class:`TemporalModule`.
"""

from .motion_module import TemporalModule

__all__ = ["TemporalModule"]
