"""Image operations. Every public op has the same signature:

    apply(img, amount, **opts) -> img

Float32 in, float32 out, no side effects. amount=0 (or default) is identity.
"""

from __future__ import annotations
