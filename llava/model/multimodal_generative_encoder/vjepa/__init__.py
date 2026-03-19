"""
Compatibility shim for vendored VJEPA package.

Upstream VJEPA code uses absolute imports like `from vjepa.xxx import yyy`.
Inside this repo it is vendored under:
`llava.model.multimodal_generative_encoder.vjepa`.
Expose an alias in `sys.modules` so those absolute imports resolve.
"""

import sys


# Keep upstream absolute-import style working for vendored package.
sys.modules.setdefault("vjepa", sys.modules[__name__])
