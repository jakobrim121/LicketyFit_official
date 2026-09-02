"""Load a public launcher's implementation into its configuration module.

The two public ``run_*.py`` files are intentionally editable configuration
surfaces.  Their validation and execution code lives beside them in private
implementation modules, but is executed in the public module namespace so the
long-standing import API (including diagnostic helpers) remains compatible.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType


_IMPLEMENTATIONS = MappingProxyType(
    {
        "wcsim": "_run_wcsim_impl.py",
        "wcte": "_run_wcte_impl.py",
    }
)


def install_launcher(namespace: dict[str, object], kind: str) -> None:
    """Execute one trusted sibling implementation in ``namespace``."""
    try:
        filename = _IMPLEMENTATIONS[str(kind)]
    except KeyError as exc:
        raise ValueError(f"unknown launcher kind {kind!r}") from exc
    path = Path(__file__).resolve().with_name(filename)
    source = path.read_bytes()
    exec(compile(source, str(path), "exec"), namespace, namespace)
