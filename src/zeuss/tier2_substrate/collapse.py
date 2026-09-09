"""Frontier 1 - entropy-driven dimensional collapse.

The logic is not a rule being evaluated; the logic *is* the structural collapse
of the space. We model a state ``z`` by how it distributes across a codebook of
atomic symbols. High entropy = ambiguous, continuous, high-dimensional. As
evidence accumulates the distribution sharpens (entropy -> 0) and the state
crystallises onto a single orthogonal symbol - a Boolean/discrete outcome.

Temperature is the knob: ``inverse_temperature`` (beta) high -> crisp/discrete,
low -> smeared/probabilistic. This is one continuum, not two modes.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..backend import HAS_JAX, RDTYPE, xp
from .hypervectors import Codebook, bundle, normalize


def softmax(x) -> "xp.ndarray":
    """Softmax over the last axis.

    ``axis=-1, keepdims=True`` is a strict generalisation, not a behaviour
    change: for the 1-D inputs every existing call site uses, the last axis
    *is* the only axis, so this returns bit-identical results to the old
    global ``max``/``sum``. It also makes this function directly reusable for
    a *batch* of distributions (shape ``(N, K)``, softmax independently per
    row) - which a global ``max``/``sum`` would get wrong, silently
    normalising the whole batch together instead of row-by-row. See
    :func:`_collapse_batch_core`, which relies on exactly this.
    """
    x = xp.asarray(x, dtype=RDTYPE)
    x = x - xp.max(x, axis=-1, keepdims=True)
    e = xp.exp(x)
    return e / xp.sum(e, axis=-1, keepdims=True)


def entropy(probs, base: float = 2.0) -> float:
    """Shannon entropy of a probability vector (bits by default)."""
    p = xp.asarray(probs, dtype=RDTYPE)
    # Mask zeros with xp.where (rather than boolean indexing) so the reduction
    # stays a fixed shape and is safe under jit/vmap: 0 * log(1) == 0.
    safe = xp.where(p > 0, p, 1.0)
    return float(-xp.sum(p * (xp.log(safe) / np.log(base))))


def occupancy(codebook: Codebook, z, inverse_temperature: float = 1.0) -> "xp.ndarray":
    """Probability the state occupies each symbol basin (a softmax over sims)."""
    sims = xp.asarray(list(codebook.similarities(z).values()), dtype=RDTYPE)
    return softmax(inverse_temperature * sims)


def collapse(codebook: Codebook, z, inverse_temperature: float = 8.0):
    """Collapse ``z`` toward the codebook under an inverse temperature.

    Returns ``(z_collapsed, info)`` where ``info`` carries the occupancy
    distribution, its entropy, and the winning symbol. As
    ``inverse_temperature`` grows, ``z_collapsed`` approaches a single symbol
    (the deterministic limit); as it shrinks, it stays a soft superposition.
    """
    names = codebook.names()
    probs = occupancy(codebook, z, inverse_temperature)
    mats = codebook.matrix()
    z_soft = bundle(list(mats), weights=list(probs))
    k = int(xp.argmax(probs))
    info = {
        "names": names,
        "probs": probs,
        "entropy_bits": entropy(probs),
        "winner": names[k],
        "winner_prob": float(probs[k]),
    }
    return normalize(z_soft), info


def _collapse_batch_core(zs, mats, inverse_temperature):
    """Pure array restatement of :func:`collapse`'s math, batched over an
    ``N``-probe leading axis instead of one probe at a time.

    Returns arrays only (``z_soft``, ``probs``, ``entropy_bits``,
    ``winner_idx``) - no Python dict, no name lookups, no ``float()`` casts -
    so, unlike ``collapse`` itself, this is safe to trace with ``jax.jit``
    (see :func:`collapse_batch_jit`). The batching is what a
    ``jax.vmap(collapse)`` would give you by auto-batching the single-probe
    function; it's written directly as vectorized array ops instead, since
    NumPy has no ``vmap`` equivalent and this vectorized form runs correctly
    on *either* backend unchanged (the ``xp`` purity convention).

    ``sims[n, k]`` is exactly ``hypervectors.similarity(zs[n], mats[k])``,
    computed for every probe against every codebook symbol in one matrix
    product instead of ``similarity``'s per-pair ``vdot`` (which only
    accepts one vector on each side).
    """
    dim = mats.shape[1]
    sims = xp.real(zs @ xp.conj(mats).T) / dim
    probs = softmax(inverse_temperature * sims)  # (N, K), independent per row
    safe = xp.where(probs > 0, probs, 1.0)
    entropy_bits = -xp.sum(probs * (xp.log(safe) / np.log(2.0)), axis=-1)  # (N,)
    winner_idx = xp.argmax(probs, axis=-1)  # (N,)
    z_soft = normalize(xp.asarray(probs, dtype=RDTYPE) @ mats)  # (N, D)
    return z_soft, probs, entropy_bits, winner_idx


def _collapse_batch_finish(codebook: Codebook, core_result) -> tuple:
    """Shared Python-side finish for collapse_batch/collapse_batch_jit: turn
    the pure-array core result into the same kind of info dict collapse()
    returns, just batched. Kept separate from _collapse_batch_core because
    name lookups and Python-side indexing can't happen inside a jax.jit trace.
    """
    z_soft, probs, entropy_bits, winner_idx = core_result
    names = codebook.names()
    winner_idx_np = np.asarray(winner_idx)
    winner_prob = xp.take_along_axis(probs, winner_idx[:, None], axis=-1)[:, 0]
    info = {
        "names": names,
        "probs": probs,
        "entropy_bits": entropy_bits,
        "winner": [names[int(k)] for k in winner_idx_np],
        "winner_prob": winner_prob,
    }
    return z_soft, info


def collapse_batch(codebook: Codebook, zs, inverse_temperature: float = 8.0):
    """Batched :func:`collapse`: resolve many probes against the same
    codebook in one vectorized pass instead of a Python loop calling
    ``collapse`` once per probe. Runs on either backend (NumPy or JAX,
    eagerly - see :func:`collapse_batch_jit` for the JIT-compiled variant).

    ``zs``: a sequence of ``N`` probe hypervectors, each shape ``(D,)``.
    Returns ``(z_collapsed, info)`` like :func:`collapse`, but every field of
    ``info`` is now batched: ``probs`` is ``(N, K)``, ``entropy_bits`` and
    ``winner_prob`` are ``(N,)``, ``winner`` is a length-``N`` list of names.
    """
    mats = codebook.matrix()
    zs_stacked = xp.stack([normalize(z) for z in zs], axis=0)
    core_result = _collapse_batch_core(zs_stacked, mats, inverse_temperature)
    return _collapse_batch_finish(codebook, core_result)


_jit_collapse_batch_core = None
if HAS_JAX:
    import jax as _jax

    # Created once, at import time, and reused for every call - jax.jit's
    # compilation cache is keyed to this specific wrapped-function object, so
    # creating a fresh jax.jit(...) wrapper inside collapse_batch_jit on every
    # call would silently retrace (and thus never actually benefit from JIT
    # compilation) instead of hitting the cache on repeated same-shape calls.
    _jit_collapse_batch_core = _jax.jit(_collapse_batch_core)


def collapse_batch_jit(codebook: Codebook, zs, inverse_temperature: float = 8.0):
    """Like :func:`collapse_batch`, but the core batched computation runs
    through a ``jax.jit``-compiled call - the concrete "jit" half of this
    roadmap item (see :func:`_collapse_batch_core` for the "vmap-style"
    batching half). XLA compiles the core once per distinct input shape
    (cached module-wide, see ``_jit_collapse_batch_core``) and reuses that
    compiled version on every later call with matching shapes.

    Requires the JAX backend; raises ``RuntimeError`` otherwise (mirroring
    :func:`zeuss.tier2_substrate.energy.settle_grad` - there is no
    meaningful JIT compilation on plain NumPy).
    """
    if not HAS_JAX:
        raise RuntimeError(
            "collapse_batch_jit requires the JAX backend - install the 'jax' extra "
            "(pip install 'zeuss[jax]') and leave ZEUSS_BACKEND unset or set it to 'jax'"
        )
    mats = codebook.matrix()
    zs_stacked = xp.stack([normalize(z) for z in zs], axis=0)
    core_result = _jit_collapse_batch_core(zs_stacked, mats, inverse_temperature)
    return _collapse_batch_finish(codebook, core_result)


def participation_ratio(probs) -> float:
    """Inverse Simpson index: the *effective* number of symbols a distribution
    is spread across, in ``[1, K]`` for ``K`` symbols.

    ``1 / sum(p_i^2)``: exactly ``1.0`` for a one-hot distribution (fully
    collapsed onto a single symbol), exactly ``K`` for the uniform
    distribution (spread evenly across all ``K``), and continuous in between
    - unlike entropy (bits, unbounded log scale), this is directly a *count*
    of live dimensions, which is what :func:`dimensional_collapse` truncates
    the basis to.
    """
    p = xp.asarray(probs, dtype=RDTYPE)
    return float(1.0 / xp.sum(p * p))


def dimensional_collapse(codebook: Codebook, z, inverse_temperature: float = 8.0):
    """Collapse ``z``, but literally shrink the live ambient dimensionality
    with entropy instead of only reweighting a fixed-size basis (that's what
    plain :func:`collapse` already does).

    The codebook's ``K`` symbols are an orthogonal basis (by construction:
    independent random phasors are quasi-orthogonal in high-D, and
    :func:`participation_ratio` treats them as exactly orthogonal, which is
    the "Boolean lattice" limit `VISION.md` describes). Only the top
    ``ceil(participation_ratio(probs))`` symbols, ranked by occupancy
    probability, are kept in the returned state - the rest are dropped from
    the basis entirely, not merely down-weighted toward zero. At maximal
    entropy (uniform occupancy) every symbol stays live and this is
    numerically identical to :func:`collapse`; as entropy drops toward zero
    the live set shrinks until only the winning symbol remains - a genuine
    ``K``-dimensional-basis to `1`-dimensional-point contraction, not a
    fixed-``D`` reweighting.
    """
    names = codebook.names()
    probs = occupancy(codebook, z, inverse_temperature)
    eff_dim = participation_ratio(probs)
    # ceil with a small epsilon tolerance: floating-point noise can push
    # eff_dim a hair above an exact integer (e.g. 1.0000000000000862 for a
    # true one-hot distribution), which plain ceil would round up to the
    # *next* integer instead of recognising it as that integer.
    k_live = max(1, int(np.ceil(eff_dim - 1e-9)))

    mats = codebook.matrix()
    order = xp.argsort(-probs)
    live_idx = order[:k_live]
    live_probs = probs[live_idx]
    live_probs = live_probs / xp.sum(live_probs)
    z_soft = bundle([mats[i] for i in live_idx], weights=list(live_probs))

    k = int(xp.argmax(probs))
    info = {
        "names": names,
        "probs": probs,
        "entropy_bits": entropy(probs),
        "winner": names[k],
        "winner_prob": float(probs[k]),
        "eff_dim": eff_dim,
        "k_live": k_live,
        "live_names": [names[int(i)] for i in live_idx],
    }
    return normalize(z_soft), info


def train_codebook(
    codebook: Codebook,
    records: Sequence[Sequence[tuple[str, str]]],
    steps: int = 150,
    learning_rate: float = 0.5,
    inverse_temperature: float = 8.0,
) -> list[float]:
    """Train a Codebook's symbol *phases* via ``jax.grad`` so role/filler
    structure in ``records`` becomes reliably recoverable - the v0.3 roadmap
    item ("train symbols so real structure self-organises").

    Each element of ``records`` is a list of ``(role, filler)`` name pairs -
    the same shape :func:`hypervectors.encode_record` consumes - describing
    one bundled record. For every pair in every record, training minimises
    the cross-entropy of unbinding that *whole bundled record* by role and
    reading occupancy against the codebook, with filler as the target -
    reusing collapse's own softmax readout as the training signal (the same
    ``inverse_temperature``-scaled similarity softmax :func:`occupancy`
    computes) rather than inventing a bespoke loss.

    This deliberately trains against the *bundled* failure mode, not an
    isolated ``bind``/``unbind`` pair: a lone pair is exactly invertible by
    phase subtraction regardless of dimension or training, so it would never
    give training anything to fix. Interference from the *other* pairs
    bundled into the same record is the real, dimension-dependent source of
    error training can reduce - confirmed empirically: at low dimension
    relative to codebook size (``dim=12``, 24 symbols, 5-pair records),
    argmax recovery accuracy starts at ~49% (near chance) and reaches 100%
    within ~40 steps of training, with the mean cross-entropy loss dropping
    from ~1.5 to ~0.13 (`test_train_codebook_improves_bundled_recovery_
    accuracy`).

    Like :func:`hypervectors.random_hypervector`, states here are
    reparameterised as real phase angles ``theta`` (``symbol = exp(i*theta)``)
    so ordinary real-to-real ``jax.grad`` applies directly and every trained
    symbol stays exactly unit-modulus by construction - the same trick
    :func:`zeuss.tier2_substrate.energy.settle_grad` uses, and for the same
    reason: :func:`occupancy`/:func:`collapse` go through Python dict lookups
    and explicit ``float()`` casts that would abort a JAX trace, so this
    carries its own self-contained, differentiable restatement of exactly
    the same bind + bundle + unbind + softmax-cross-entropy math instead.

    Mutates ``codebook``'s stored symbols in place (via ``Codebook.add``) and
    returns the per-step loss trace (length ``steps + 1``, including the
    pre-training loss at index 0). Requires the JAX backend; raises
    ``RuntimeError`` otherwise (mirroring ``settle_grad``/``collapse_batch_jit``
    - there is no meaningful autodiff fallback on plain NumPy).
    """
    if not HAS_JAX:
        raise RuntimeError(
            "train_codebook requires the JAX backend - install the 'jax' extra "
            "(pip install 'zeuss[jax]') and leave ZEUSS_BACKEND unset or set it to 'jax'"
        )
    import jax

    names = codebook.names()
    name_idx = {name: i for i, name in enumerate(names)}
    dim = codebook.dim
    record_role_idx = [xp.asarray([name_idx[r] for r, _ in rec]) for rec in records]
    record_filler_idx = [xp.asarray([name_idx[f] for _, f in rec]) for rec in records]

    def loss_of_theta(theta):
        mats = xp.exp(1j * theta)  # (K, D), exactly unit-modulus by construction
        logits_parts = []
        target_parts = []
        for role_idx, filler_idx in zip(record_role_idx, record_filler_idx):
            roles = mats[role_idx]  # (P, D)
            fillers = mats[filler_idx]  # (P, D)
            bound = roles * fillers  # bind, still unit-modulus
            summed = xp.sum(bound, axis=0)  # bundle: (D,)
            record_vec = summed / xp.abs(summed)  # normalize
            probes = record_vec[None, :] * xp.conj(roles)  # unbind each role: (P, D)
            sims = xp.real(probes @ xp.conj(mats).T) / dim  # (P, K)
            logits_parts.append(inverse_temperature * sims)
            target_parts.append(filler_idx)
        logits = xp.concatenate(logits_parts, axis=0)  # (N, K)
        targets = xp.concatenate(target_parts, axis=0)  # (N,)
        # log-sum-exp, the same max-subtraction stability trick softmax() uses.
        m = xp.max(logits, axis=-1, keepdims=True)
        log_z = xp.log(xp.sum(xp.exp(logits - m), axis=-1)) + m[:, 0]
        target_logits = logits[xp.arange(logits.shape[0]), targets]
        return xp.mean(log_z - target_logits)

    grad_fn = jax.grad(loss_of_theta)
    effective_lr = learning_rate * dim  # same scaling convention as settle_grad

    theta = xp.angle(codebook.matrix())
    losses = [float(loss_of_theta(theta))]
    for _ in range(steps):
        theta = theta - effective_lr * grad_fn(theta)
        losses.append(float(loss_of_theta(theta)))

    trained = xp.exp(1j * theta)
    for i, name in enumerate(names):
        codebook.add(name, trained[i])
    return losses


_ANNEAL_TRACE_KEYS = ("entropy_bits", "winner", "winner_prob", "eff_dim", "k_live")


def anneal(codebook: Codebook, z, schedule=(0.5, 1, 2, 4, 8, 16, 32)):
    """Run a cooling schedule and record entropy dropping toward a decision.

    Returns a list of per-step info dicts - a trace of continuous ambiguity
    crystallising into a discrete symbol. Uses :func:`dimensional_collapse`
    rather than plain :func:`collapse` so ``eff_dim``/``k_live`` - Frontier
    1's own "effective dimension" primitive, previously computed nowhere in
    this pipeline - become part of that same trace: the live basis actually
    shrinking as beta rises, not just entropy_bits falling. ``z`` is fixed
    throughout (only beta changes), so this is a pure addition: at high
    entropy ``dimensional_collapse`` is numerically identical to ``collapse``
    (checked directly - see ``test_dimensional_collapse_matches_collapse_at_
    high_entropy``), and neither function's returned state feeds into the
    next step here, so every existing ``entropy_bits``/``winner``/
    ``winner_prob`` value in the trace is unchanged.
    """
    trace = []
    for beta in schedule:
        _, info = dimensional_collapse(codebook, z, inverse_temperature=beta)
        trace.append({"beta": float(beta), **{k: info[k] for k in _ANNEAL_TRACE_KEYS}})
    return trace


def anneal_adaptive(
    codebook: Codebook,
    z,
    beta_start: float = 0.5,
    beta_max: float = 64.0,
    entropy_tol: float = 0.05,
    growth_range: tuple[float, float] = (1.05, 3.0),
    margin_scale: float = 0.5,
    max_steps: int = 40,
):
    """Cooling schedule whose beta growth rate adapts to the probe's ambiguity.

    "Liquid time-step" applied to Frontier 1. ``z`` is fixed throughout a
    schedule - only beta changes - so the probe's intrinsic hardness (the raw
    similarity gap between its best and second-best codebook symbol) is a
    single beta-independent number, computed once up front: a small gap (two
    close, competing symbols - a genuinely hard decision) grows beta slowly,
    spending more of the schedule resolving the ambiguity; a wide gap
    (unambiguous probe) grows beta quickly, reaching the discrete decision in
    fewer steps than a fixed static schedule needs. Stops once entropy drops
    below ``entropy_tol`` or ``beta_max``/``max_steps`` is hit.

    Each trace entry also carries ``eff_dim``/``k_live`` (see :func:`anneal`
    for why this now uses :func:`dimensional_collapse` instead of plain
    :func:`collapse`) - the live basis shrinking in lockstep with the
    adaptive schedule's own beta growth.
    """
    sims = sorted(codebook.similarities(z).values(), reverse=True)
    margin = (sims[0] - sims[1]) if len(sims) > 1 else 1.0
    lo, hi = growth_range
    growth = lo + (hi - lo) * float(np.clip(margin / margin_scale, 0.0, 1.0))

    beta = beta_start
    trace = []
    for _ in range(max_steps):
        _, info = dimensional_collapse(codebook, z, inverse_temperature=beta)
        trace.append({"beta": float(beta), **{k: info[k] for k in _ANNEAL_TRACE_KEYS}})
        if info["entropy_bits"] <= entropy_tol or beta >= beta_max:
            break
        beta = min(beta * growth, beta_max)
    return trace
