#!/usr/bin/env python3
"""Run ``batch_fit_driver`` as ``__main__`` through the import system.

CPython never writes a ``.pyc`` for the script named on the command line, so
launching the 44k-line driver directly re-compiles it from source on every
process start (~0.2-0.4 s).  Running it with ``runpy.run_module`` goes through
the normal loader, which reads and maintains ``scripts/__pycache__``.

Semantics are otherwise unchanged: the driver still sees ``__name__ ==
"__main__"``, its own ``__file__``, ``sys.argv`` as given, and ``sys.path[0]``
equal to this ``scripts`` directory exactly as when run as a script.
"""
import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if sys.path[0] != _HERE:
    sys.path.insert(0, _HERE)

runpy.run_module("batch_fit_driver", run_name="__main__", alter_sys=True)
