"""Objective-only convergence control for exact block optimizers.

The controller in this module does not know detector geometry, particle type,
or likelihood details.  It only observes exact-objective decreases reported by
the unchanged optimizer and requests at most one additional sweep at a time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Callable


@dataclass(frozen=True)
class AdaptiveExactDiagnostics:
    """Serializable convergence and work accounting for one exact fit."""

    enabled: bool
    status: str
    converged: bool
    cap_reached: bool
    base_sweeps: int
    extra_sweeps: int
    total_sweeps: int
    attempted_extra_sweeps: int
    accepted_extra_sweeps: int
    initial_gain_nll: float
    final_gain_nll: float
    convergence_threshold_nll: float
    max_total_sweeps: int
    backend: str
    backend_diagnostics: dict[str, Any] | None = None
    failure: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _last_sweep_gain(history) -> float:
    """Read the last exact sweep's gain, including a one-sweep history."""

    if not history:
        return math.nan
    final = history[-1]
    if "sweep_gain" in final:
        return float(final["sweep_gain"])
    if "sweep_start_fval" in final and "fval" in final:
        return float(final["sweep_start_fval"]) - float(final["fval"])
    if len(history) >= 2:
        return float(history[-2]["fval"]) - float(final["fval"])
    return math.nan


def _merge_accepted_continuation(current, continuation, *, index: int, gain: float):
    """Accumulate work/history after an exact one-sweep continuation is accepted."""

    continuation.nfcn += int(current.nfcn)
    continuation.wall_s += float(current.wall_s)
    continuation.invalid_evaluations += int(current.invalid_evaluations)
    continuation.quadratic_skips += int(current.quadratic_skips)
    base_count = int(len(current.history))
    continuation.history = list(current.history) + [
        {
            **entry,
            "adaptive_polish": True,
            "adaptive_polish_index": int(index),
            "adaptive_polish_gain": float(gain),
            "adaptive_total_sweep_index": int(base_count + offset),
        }
        for offset, entry in enumerate(continuation.history)
    ]
    return continuation


def continue_adaptive_exact(
    result,
    *,
    enabled: bool,
    convergence_threshold_nll: float,
    max_total_sweeps: int,
    continue_step: Callable[[Any], Any],
    backend: str = "one_sweep_block",
    acceptance_epsilon_nll: float = 1.0e-10,
    continuation_units: int = 1,
):
    """Continue an exact result one sweep at a time until converged or capped.

    ``continue_step`` must start from the supplied result and return an
    incremental continuation result. The controller is backend-agnostic: the
    standard fallback requests one carried-radius block sweep, while a certified
    six-coordinate derivative-free continuation can be injected without changing
    the stopping/accounting policy. Only a finite, strictly lower exact objective
    is accepted. A converged input is returned by identity and the callback is
    never evaluated.
    """

    threshold = float(convergence_threshold_nll)
    cap = int(max_total_sweeps)
    epsilon = float(acceptance_epsilon_nll)
    units = int(continuation_units)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("convergence_threshold_nll must be finite and nonnegative")
    if cap < 1:
        raise ValueError("max_total_sweeps must be positive")
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("acceptance_epsilon_nll must be finite and nonnegative")
    if units < 1:
        raise ValueError("continuation_units must be positive")

    base_sweeps = int(len(result.history))
    initial_gain = _last_sweep_gain(result.history)

    def diagnostics(
        status,
        *,
        final_gain=initial_gain,
        extra=0,
        attempted=0,
        accepted=0,
        converged=False,
        cap_reached=False,
        failure=None,
        backend_diagnostics=None,
    ):
        return AdaptiveExactDiagnostics(
            enabled=bool(enabled),
            status=str(status),
            converged=bool(converged),
            cap_reached=bool(cap_reached),
            base_sweeps=base_sweeps,
            extra_sweeps=int(extra),
            total_sweeps=int(base_sweeps + extra),
            attempted_extra_sweeps=int(attempted),
            accepted_extra_sweeps=int(accepted),
            initial_gain_nll=float(initial_gain),
            final_gain_nll=float(final_gain),
            convergence_threshold_nll=threshold,
            max_total_sweeps=cap,
            backend=str(backend),
            backend_diagnostics=(
                None if backend_diagnostics is None else dict(backend_diagnostics)
            ),
            failure=None if failure is None else str(failure),
        )

    if not enabled:
        return result, diagnostics("disabled")
    if not math.isfinite(initial_gain):
        return result, diagnostics(
            "indeterminate_sweep_gain",
            failure="exact optimizer history does not report a finite final sweep gain",
        )
    if initial_gain <= threshold:
        return result, diagnostics("converged", converged=True)
    if base_sweeps >= cap:
        return result, diagnostics("cap_reached", cap_reached=True)

    current = result
    final_gain = initial_gain
    extra_sweeps = 0
    attempted = 0
    accepted = 0
    while final_gain > threshold and base_sweeps + extra_sweeps < cap:
        attempted += 1
        try:
            continuation_payload = continue_step(current)
        except Exception as exc:
            return current, diagnostics(
                "continuation_failure",
                final_gain=final_gain,
                extra=extra_sweeps,
                attempted=attempted,
                accepted=accepted,
                failure=repr(exc),
            )

        if (
            isinstance(continuation_payload, tuple)
            and len(continuation_payload) == 2
            and isinstance(continuation_payload[1], dict)
        ):
            continuation, backend_diagnostics = continuation_payload
        else:
            continuation = continuation_payload
            backend_diagnostics = None

        continuation_history = list(getattr(continuation, "history", ()))
        if len(continuation_history) != 1:
            return current, diagnostics(
                "invalid_continuation_history",
                final_gain=final_gain,
                extra=extra_sweeps,
                attempted=attempted,
                accepted=accepted,
                failure=(
                    "one-sweep continuation returned "
                    f"{len(continuation_history)} history entries"
                ),
                backend_diagnostics=backend_diagnostics,
            )

        extra_sweeps += units
        candidate_fval = float(continuation.fval)
        gain = float(current.fval) - candidate_fval
        final_gain = gain
        if not math.isfinite(candidate_fval) or not math.isfinite(gain):
            return current, diagnostics(
                "nonfinite_continuation",
                final_gain=final_gain,
                extra=extra_sweeps,
                attempted=attempted,
                accepted=accepted,
                failure="one-sweep continuation returned a non-finite objective",
                backend_diagnostics=backend_diagnostics,
            )
        if gain <= epsilon:
            return current, diagnostics(
                "converged_no_accepted_improvement",
                final_gain=final_gain,
                extra=extra_sweeps,
                attempted=attempted,
                accepted=accepted,
                converged=True,
                backend_diagnostics=backend_diagnostics,
            )

        current = _merge_accepted_continuation(
            current,
            continuation,
            index=accepted,
            gain=gain,
        )
        accepted += 1
        if backend_diagnostics is not None and bool(
            backend_diagnostics.get("converged", False)
        ):
            return current, diagnostics(
                "converged",
                final_gain=final_gain,
                extra=extra_sweeps,
                attempted=attempted,
                accepted=accepted,
                converged=True,
                backend_diagnostics=backend_diagnostics,
            )

    cap_reached = bool(final_gain > threshold)
    return current, diagnostics(
        "cap_reached" if cap_reached else "converged",
        final_gain=final_gain,
        extra=extra_sweeps,
        attempted=attempted,
        accepted=accepted,
        converged=not cap_reached,
        cap_reached=cap_reached,
        backend_diagnostics=(
            backend_diagnostics if "backend_diagnostics" in locals() else None
        ),
    )
