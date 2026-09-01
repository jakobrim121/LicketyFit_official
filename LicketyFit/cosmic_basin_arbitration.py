"""Truth-blind discrete-basin arbitration for cosmic MCS continuations."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from typing import Any


def select_coherent_basin_payloads(
    optimized_basins: Iterable[Mapping[str, Any]],
    chosen_fit: Mapping[str, Any],
    *,
    start_hypothesis: str,
    score: Callable[[Mapping[str, Any]], float],
    max_candidates: int,
    gate_nll: float,
) -> tuple[Mapping[str, Any], ...]:
    """Return distinct, nearly tied line basins for coherent-MCS profiling.

    The ordinary cosmic tournament remains authoritative for the discrete
    start topology. Within that smooth topology, a straight-line optimum is a
    proposal rather than a final model decision: the scattered-track
    likelihood must be allowed to arbitrate between nearly tied line basins.

    The best finalized payload per seed is retained, candidates outside the
    straight-model NLL gate are discarded, and no event-truth quantity enters
    either the filtering or ordering.
    """

    maximum = max(1, int(max_candidates))
    gate = float(gate_nll)
    if not math.isfinite(gate) or gate < 0.0:
        raise ValueError("coherent basin gate must be finite and nonnegative")

    selected_hypothesis = str(start_hypothesis)
    chosen_score = float(score(chosen_fit))
    if not math.isfinite(chosen_score):
        raise ValueError("chosen coherent-basin score must be finite")

    best_by_seed: dict[int, tuple[float, Mapping[str, Any]]] = {}
    for payload in (chosen_fit, *tuple(optimized_basins)):
        if str(payload.get("track_start_hypothesis", "")) != selected_hypothesis:
            continue
        payload_score = float(score(payload))
        if not math.isfinite(payload_score):
            continue
        if payload_score > chosen_score + gate + 1.0e-12:
            continue
        seed_index = int(payload.get("seed_index", -1))
        incumbent = best_by_seed.get(seed_index)
        if incumbent is None or payload_score < incumbent[0]:
            best_by_seed[seed_index] = (payload_score, payload)

    if not best_by_seed:
        return (chosen_fit,)
    ranked = sorted(
        best_by_seed.values(),
        key=lambda item: (
            float(item[0]), int(item[1].get("seed_index", -1))
        ),
    )
    return tuple(payload for _, payload in ranked[:maximum])
