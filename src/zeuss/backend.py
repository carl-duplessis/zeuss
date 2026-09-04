"""Array backend abstraction.

Zeuss is written against a single array namespace ``xp`` so the same code runs
on NumPy (default, always available) or JAX (optional, for autodiff + XLA on
GPU/TPU). Set the environment variable ``ZEUSS_BACKEND=jax`` (or ``numpy``) to
force a choice; otherwise JAX is used when importable and NumPy otherwise.

All substrate math uses complex dtypes (``complex64``/``complex128``) because
hypervectors are continuous-phase phasors z = r * e^{i*theta}.
"""
from __future__ import annotations

import os

_pref = os.environ.get("ZEUSS_BACKEND", "auto").lower()

HAS_JAX = False
xp = None

if _pref in ("auto", "jax"):
    try:
        import jax  # noqa: F401
        import jax.numpy as jnp

        xp = jnp
        HAS_JAX = True
        # 64-bit complex needs to be enabled explicitly in JAX.
        try:
            jax.config.update("jax_enable_x64", True)
        except Exception:  # pragma: no cover - defensive
            pass
    except Exception:
        if _pref == "jax":
            raise
        HAS_JAX = False

if xp is None:
    import numpy as _np

    xp = _np
    HAS_JAX = False

# Default complex/real dtypes for the substrate.
CDTYPE = xp.complex128
RDTYPE = xp.float64


def to_numpy(a):
    """Return a plain NumPy array regardless of the active backend."""
    import numpy as _np

    if HAS_JAX:
        return _np.asarray(a)
    return _np.asarray(a)


def backend_name() -> str:
    return "jax" if HAS_JAX else "numpy"
