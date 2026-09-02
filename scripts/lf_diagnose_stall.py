#!/usr/bin/env python3
"""Diagnose a stalled LicketyFit multiprocess run WITHOUT ptrace.

Run this in a second terminal on the SAME node while the run is stalled:

    python3 lf_diagnose_stall.py            # auto-find your LicketyFit processes
    python3 lf_diagnose_stall.py --pid N    # or give the launcher/chunk PID

It reads only /proc (no gdb, no py-spy needed) and answers, per worker:
  * is it alive, and is it BURNING CPU (R, cpu-seconds advancing) or BLOCKED
    (S/D, cpu not advancing) — this alone separates "compiling/thrashing"
    from "deadlocked on a lock/pipe";
  * what kernel wait channel it is in (wchan), and its thread count;
  * RSS per worker, the total, and the cgroup memory limit/usage/oom-kills
    — 16 fitter workers can exceed an lxplus per-user memory limit, which
    presents as a silent crawl;
  * file locks it holds or waits on (/proc/locks), and open files under the
    runtime cache / staging directories — a shared-file contention deadlock
    shows up here;
  * a Python stack via py-spy IF it is installed and permitted (best effort).

It also prints the retained chunk log tail and the runtime-stage contents so
you can see whether workers are missing cache payloads.

Nothing here modifies the run.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HZ = os.sysconf("SC_CLK_TCK")
PAGE = os.sysconf("SC_PAGE_SIZE")


def read(path, default=""):
    try:
        return Path(path).read_text()
    except Exception:
        return default


def cmdline(pid):
    return read(f"/proc/{pid}/cmdline").replace("\0", " ").strip()


_LF_SCRIPTS = ("batch_fit_driver", "lf_driver_main", "run_wcsim", "run_wcte",
               "lf_bisect_np")


def is_licketyfit(pid):
    """True only for a Python process actually *running* a LicketyFit script.

    Matching the raw command line would also catch the shell that created the
    file and this diagnostic itself, so require a python interpreter and a
    matching .py argument.
    """
    c = cmdline(pid)
    if "lf_diagnose_stall" in c:
        return False
    args = c.split()
    if not args or "python" not in os.path.basename(args[0]):
        return False
    return any(
        any(k in os.path.basename(a) for k in _LF_SCRIPTS)
        for a in args[1:] if a.endswith(".py")
    )


def ppid_of(pid):
    for line in read(f"/proc/{pid}/status").splitlines():
        if line.startswith("PPid:"):
            return int(line.split()[1])
    return None


def all_pids():
    return [int(p) for p in os.listdir("/proc") if p.isdigit()]


def descendants(root):
    kids = {}
    for p in all_pids():
        pp = ppid_of(p)
        if pp is not None:
            kids.setdefault(pp, []).append(p)
    out, stack = [], [root]
    while stack:
        p = stack.pop()
        for k in kids.get(p, []):
            out.append(k)
            stack.append(k)
    return sorted(out)


def stat_fields(pid):
    s = read(f"/proc/{pid}/stat")
    if not s:
        return None
    # comm may contain spaces; split after the last ')'
    rest = s[s.rindex(")") + 2:].split()
    return {
        "state": rest[0],
        "utime": int(rest[11]) / HZ,
        "stime": int(rest[12]) / HZ,
        "threads": int(rest[17]),
        "rss_mb": int(rest[21]) * PAGE / 1e6,
    }


def smaps_rollup(pid):
    """Return (rss_mb, pss_mb) from /proc/PID/smaps_rollup.

    RSS counts every mapped page in full, so summing it over forked workers
    counts each copy-on-write page once per process and can overstate the
    real cost several-fold.  PSS divides each shared page by the number of
    sharers, so the PSS sum is the actual physical footprint.
    """
    rss = pss = 0.0
    try:
        for line in read(f"/proc/{pid}/smaps_rollup").splitlines():
            key, _, val = line.partition(":")
            if key == "Rss":
                rss = int(val.split()[0]) / 1024
            elif key == "Pss":
                pss = int(val.split()[0]) / 1024
    except Exception:
        pass
    return rss, pss


def wchan(pid):
    return read(f"/proc/{pid}/wchan").strip() or "?"


def ctx_switches(pid):
    v = {}
    for line in read(f"/proc/{pid}/status").splitlines():
        if "ctxt_switches" in line:
            k, n = line.split(":")
            v[k.strip()] = int(n)
    return v


def open_files(pid, roots):
    hits = []
    try:
        for fd in os.listdir(f"/proc/{pid}/fd"):
            try:
                target = os.readlink(f"/proc/{pid}/fd/{fd}")
            except OSError:
                continue
            if any(target.startswith(r) for r in roots):
                hits.append(target)
    except OSError:
        pass
    return hits


def locks_for(pids):
    """Parse /proc/locks: returns {pid: [lines]} for the given pids."""
    out = {}
    for line in read("/proc/locks").splitlines():
        parts = line.split()
        # format: id: [->] TYPE MODE ACCESS PID dev:inode start end
        try:
            pid = int(parts[4 if parts[1] != "->" else 5])
        except (IndexError, ValueError):
            continue
        if pid in pids:
            waiting = parts[1] == "->"
            out.setdefault(pid, []).append(("WAITING" if waiting else "HELD") + ": " + line)
    return out


def cgroup_memory():
    """Return (limit_bytes, usage_bytes, oom_kills) for cgroup v2 or v1."""
    try:
        cg = read("/proc/self/cgroup").strip().splitlines()[-1].split(":")[-1]
    except Exception:
        return None
    v2 = Path("/sys/fs/cgroup") / cg.lstrip("/")
    if (v2 / "memory.max").exists():
        limit = read(v2 / "memory.max").strip()
        usage = read(v2 / "memory.current").strip()
        events = read(v2 / "memory.events")
        oom = sum(int(l.split()[1]) for l in events.splitlines() if l.startswith("oom_kill"))
        return ("v2", str(v2), limit, usage, oom)
    v1 = Path("/sys/fs/cgroup/memory") / cg.lstrip("/")
    if (v1 / "memory.limit_in_bytes").exists():
        return ("v1", str(v1),
                read(v1 / "memory.limit_in_bytes").strip(),
                read(v1 / "memory.usage_in_bytes").strip(),
                read(v1 / "memory.oom_control").count("oom_kill_disable 0"))
    return None


def py_spy(pid):
    exe = shutil.which("py-spy")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "dump", "--pid", str(pid), "--nonblocking"],
                           capture_output=True, text=True, timeout=20)
        return (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        return f"py-spy failed: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, help="launcher or chunk-child PID (default: auto)")
    ap.add_argument("--sample", type=float, default=5.0, help="seconds between CPU samples")
    ap.add_argument("--stage", default=f"/tmp/{os.environ.get('USER','')}",
                    help="node-local runtime stage root")
    args = ap.parse_args()

    import socket
    host = socket.gethostname()
    print(f"host: {host}    (lxplus is load-balanced: /proc is per-node --")
    print(f"                 this MUST be the same node the stalled run is on)\n")

    me = os.getpid()
    if args.pid:
        roots = [args.pid]
    else:
        roots = [p for p in all_pids() if p != me and is_licketyfit(p) and
                 not is_licketyfit(ppid_of(p) or 0)]
    if not roots:
        print("No LicketyFit processes found on this node.")
        print("  * If your run is on a different lxplus node, ssh to THAT node")
        print("    (e.g. `ssh lxplus812`) and rerun this script there.")
        print("  * Otherwise pass --pid <launcher pid>.")
        return 1

    # Distinguish a live run from orphaned leftovers of previous runs.
    live, orphan = [], []
    for r in roots:
        (orphan if ppid_of(r) == 1 else live).append(r)
    if orphan:
        print("WARNING: these processes have ppid=1, i.e. their launcher is gone.")
        print("They are leftovers from an earlier run, NOT the live one:")
        for o in orphan:
            s = stat_fields(o)
            rss = f"{s['rss_mb']:.0f} MB" if s else "?"
            print(f"    pid {o}  rss {rss}  {cmdline(o)[:70]}")
        print("Clean them up with:  kill " + " ".join(str(o) for o in orphan))
        if not live:
            print("\nNo LIVE LicketyFit tree on this node -- see the node note above.\n")
        print()
    if live:
        roots = live

    cache_roots = [args.stage, os.path.expanduser("~/.cache/licketyfit"), "/tmp"]
    for root in roots:
        tree = [root] + descendants(root)
        print("=" * 78)
        print(f"process tree rooted at {root}: {len(tree)} processes")
        for p in tree:
            print(f"  {p:>7d}  ppid={ppid_of(p)!s:>7}  {cmdline(p)[:100]}")

        print("\n--- sampling CPU for %.0f s (does anyone make progress?) ---" % args.sample)
        s0 = {p: stat_fields(p) for p in tree}
        c0 = {p: ctx_switches(p) for p in tree}
        time.sleep(args.sample)
        s1 = {p: stat_fields(p) for p in tree}
        c1 = {p: ctx_switches(p) for p in tree}

        total_rss = 0.0
        total_pss = 0.0
        print(f"\n{'pid':>7} {'st':>2} {'cpu s/s':>8} {'thr':>4} {'RSS MB':>8} {'PSS MB':>8} {'vol/nonvol ctx sw':>18}  wchan")
        for p in tree:
            a, b = s0.get(p), s1.get(p)
            if not a or not b:
                print(f"{p:>7}  (exited during sample)")
                continue
            dcpu = (b["utime"] + b["stime"]) - (a["utime"] + a["stime"])
            vol = c1[p].get("voluntary_ctxt_switches", 0) - c0[p].get("voluntary_ctxt_switches", 0)
            nonvol = c1[p].get("nonvoluntary_ctxt_switches", 0) - c0[p].get("nonvoluntary_ctxt_switches", 0)
            rss_mb, pss_mb = smaps_rollup(p)
            if rss_mb <= 0.0:
                rss_mb = b["rss_mb"]
            total_rss += rss_mb
            total_pss += pss_mb
            verdict = ""
            if dcpu / args.sample > 0.2:
                verdict = "  <- BUSY (computing/compiling)"
            elif b["state"] == "D":
                verdict = "  <- BLOCKED IN KERNEL I/O (AFS? /tmp full?)"
            elif dcpu < 0.05:
                verdict = "  <- IDLE/BLOCKED (waiting on lock, pipe, or peer)"
            print(f"{p:>7} {b['state']:>2} {dcpu/args.sample:8.2f} {b['threads']:>4} {rss_mb:8.0f} "
                  f"{pss_mb:8.0f} {vol:>8}/{nonvol:<9} {wchan(p)}{verdict}")
        print(f"\nTOTAL: RSS {total_rss/1e3:.1f} GB (over-counts shared pages)"
              f"   PSS {total_pss/1e3:.1f} GB  <- ACTUAL physical memory used")

        print("\n--- per-user memory limit (the session scope alone is not "
              "authoritative) ---")
        uid = os.getuid()
        for base in (f"/sys/fs/cgroup/memory/user.slice/user-{uid}.slice",
                     f"/sys/fs/cgroup/user.slice/user-{uid}.slice"):
            bp = Path(base)
            if not bp.is_dir():
                continue
            for name in ("memory.limit_in_bytes", "memory.max",
                         "memory.max_usage_in_bytes", "memory.peak",
                         "memory.failcnt", "memory.events"):
                v = read(bp / name).strip()
                if v:
                    if v.isdigit() and int(v) > 1 << 30:
                        v = f"{int(v)/1e9:.1f} GB"
                    print(f"  {base}/{name}: {v.splitlines()[0] if v else v}")
        print("  (failcnt > 0, or peak == limit, means you hit the ceiling)")

        cg = cgroup_memory()
        if cg:
            kind, path, limit, usage, oom = cg
            lim = f"{int(limit)/1e9:.1f} GB" if limit.isdigit() else limit
            use = f"{int(usage)/1e9:.1f} GB" if usage.isdigit() else usage
            print(f"cgroup {kind} memory: usage {use} / limit {lim}   oom_kills={oom}   ({path})")
            if limit.isdigit() and usage.isdigit() and int(usage) > 0.9 * int(limit):
                print("  !!! at memory limit — workers are being throttled/swapped; reduce NPROC")
        else:
            print("cgroup memory info not readable")

        print("\n--- file locks held/waited by the tree (/proc/locks) ---")
        lk = locks_for(set(tree))
        if not lk:
            print("  none")
        for p, lines in lk.items():
            for l in lines:
                print(f"  {p}: {l}")

        print("\n--- open files under cache/stage roots (first 6 per process) ---")
        for p in tree:
            f = open_files(p, cache_roots)
            if f:
                print(f"  {p}: " + " | ".join(x[-70:] for x in f[:6]))

        if shutil.which("py-spy"):
            print("\n--- py-spy Python stacks (best effort) ---")
            for p in tree:
                s = stat_fields(p)
                if not s:
                    continue
                dump = py_spy(p)
                if dump:
                    print(f"\n### pid {p}")
                    print("\n".join(dump.splitlines()[:40]))
        else:
            print("\n(py-spy not installed: `pip install py-spy` then rerun for Python stacks;"
                  " needs ptrace permission — if it says 'Operation not permitted', use the"
                  " /proc evidence above and the bisect procedure instead)")

    print("\n--- runtime stage contents ---")
    stage = Path(args.stage)
    if not stage.exists():
        print(f"  {stage} does not exist on this node "
              "(another node, or the run never staged)")
    for d in sorted(stage.glob("licketyfit-*/runtime-stage/*")):
        print(f"  {d}")
        for sub in ("numba", "native", "reflection", "response", "tables", "geometry"):
            n = sum(1 for _ in (d / sub).rglob("*") if _.is_file()) if (d / sub).exists() else "MISSING"
            print(f"     {sub:11s} {n}")

    print("\n--- retained chunk logs (newest first) ---")
    logs = sorted(Path(os.path.expanduser("~")).rglob("*.parts.*/*.log"), key=os.path.getmtime, reverse=True)[:3]
    logs += sorted(Path("/tmp").rglob("*.parts.*/*.log"), key=os.path.getmtime, reverse=True)[:3]
    for lg in logs[:4]:
        print(f"\n### {lg}")
        print("\n".join(read(lg).splitlines()[-25:]))
    if not logs:
        print("  none found under ~ or /tmp (check the OUTPUT_FILE directory: <output>.parts.<pid>/*.log)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
