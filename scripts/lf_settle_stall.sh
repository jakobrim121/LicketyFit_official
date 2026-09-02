#!/bin/bash
# Settle the LicketyFit stall in ~15 minutes.
#
#   cd <package>/scripts
#   bash lf_settle_stall.sh
#
# Three steps, run automatically:
#   1. Reproduce at your configured NPROC but with a SHORT no-result timeout,
#      so instead of hanging it fails fast and NAMES the events whose worker
#      died holding them.
#   2. Collect kill evidence: kernel OOM records, per-user cgroup failcnt/peak,
#      and true physical memory (PSS, not the over-counting RSS).
#   3. Re-fit exactly those named events serially (NPROC=1).
#
# Verdict:
#   * serial re-fit SUCCEEDS  -> workers are being killed; a parallelism or
#     resource problem, not a bad event.  Step 2 says whether it was the OOM
#     killer.  If not OOM, it is a crash in native/JIT code under fork.
#   * serial re-fit CRASHES   -> an event-dependent bug; you now have the
#     event index and a real traceback instead of a silent hang.
set -u
LAUNCHER=${LAUNCHER:-run_wcsim.py}
TIMEOUT=${TIMEOUT_S:-120}
OUT=${OUTDIR:-/tmp/${USER:-$(id -un)}/lf_settle}
mkdir -p "$OUT"
cp -f "$(dirname "$0")/_mk_repro.py" "$OUT/_mk_repro.py" 2>/dev/null || true
UID_NUM=$(id -u)
echo "host $(hostname)   package $(cd .. && pwd)   logs $OUT"

[ -f "$LAUNCHER" ] || { echo "ERROR: run from the package's scripts/ directory" >&2; exit 2; }

# ---------- baseline memory before the run ----------
for base in /sys/fs/cgroup/memory/user.slice/user-$UID_NUM.slice \
            /sys/fs/cgroup/user.slice/user-$UID_NUM.slice; do
  [ -d "$base" ] || continue
  for f in memory.failcnt memory.max_usage_in_bytes memory.peak; do
    [ -r "$base/$f" ] && echo "  before: $f = $(cat "$base/$f" 2>/dev/null)"
  done
done
DMESG_OK=0; dmesg -T >/dev/null 2>&1 && DMESG_OK=1
DMESG_BEFORE=$(dmesg -T 2>/dev/null | grep -ci "killed process"); DMESG_BEFORE=${DMESG_BEFORE:-0}

# ---------- step 1: fail fast instead of hanging ----------
echo
echo "=== STEP 1: reproduce with a ${TIMEOUT}s no-result timeout ==="
# The launcher OVERWRITES LF_EVENT_RESULT_STALL_TIMEOUT_SECONDS from its own
# EVENT_RESULT_STALL_TIMEOUT_SECONDS constant, so exporting it does nothing.
# Patch that constant in a temporary launcher copy instead, and raise the
# outer chunk watchdog above it so the inner diagnostic fires first.
repro="./lf_settle_repro.py"
python3 "$OUT/_mk_repro.py" "$LAUNCHER" "$repro" "$TIMEOUT" || {
  echo "could not patch the timeout into a launcher copy"; exit 2; }
echo "  running $repro   (live: tail -f $OUT/repro.log)"
echo "  expect a verdict about $((TIMEOUT+200))s from now"
python3 "$repro" > "$OUT/repro.log" 2>&1
rc=$?
rm -f "$repro"
tail -8 "$OUT/repro.log"
UNRESOLVED=$(grep -o "unresolved source event indices: \[[^]]*\]" "$OUT/repro.log" \
             | tail -1 | sed 's/.*\[//; s/\]//')
echo
if [ -z "$UNRESOLVED" ]; then
  if [ $rc -eq 0 ]; then
    echo ">>> The run COMPLETED this time. The stall is intermittent;"
    echo ">>> keep LF_EVENT_RESULT_STALL_TIMEOUT_SECONDS set and rerun to catch it."
    exit 0
  fi
  echo ">>> No 'unresolved' line found. Full log: $OUT/repro.log"
  tail -30 "$OUT/repro.log"; exit 1
fi
echo ">>> Events whose worker died holding them: $UNRESOLVED"

# ---------- step 2: kill evidence ----------
echo
echo "=== STEP 2: was it the OOM killer? ==="
DMESG_AFTER=$(dmesg -T 2>/dev/null | grep -ci "killed process"); DMESG_AFTER=${DMESG_AFTER:-0}
if [ "$DMESG_OK" -eq 0 ]; then
  echo "    dmesg not readable on this node - use the cgroup counters below."
elif [ "$DMESG_AFTER" -gt "$DMESG_BEFORE" ]; then
  echo ">>> YES - kernel recorded new OOM kills during the run:"
  dmesg -T 2>/dev/null | grep -i "killed process" | tail -5
  echo ">>> FIX: lower NPROC, or submit to HTCondor with request_memory."
else
  echo ">>> NO new kernel OOM kills. Workers died some other way (crash)."
fi
for base in /sys/fs/cgroup/memory/user.slice/user-$UID_NUM.slice \
            /sys/fs/cgroup/user.slice/user-$UID_NUM.slice; do
  [ -d "$base" ] || continue
  for f in memory.failcnt memory.max_usage_in_bytes memory.peak \
           memory.limit_in_bytes memory.max memory.events; do
    [ -r "$base/$f" ] && echo "  after:  $f = $(head -2 "$base/$f" 2>/dev/null | tr '\n' ' ')"
  done
done
echo "  (failcnt > 0, or peak == limit, means you hit the per-user ceiling)"

# ---------- step 3: serial re-fit of the named events ----------
FIRST=$(echo "$UNRESOLVED" | tr -d ' ' | cut -d, -f1)
LAST=$(echo "$UNRESOLVED" | tr -d ' ' | tr ',' '\n' | tail -1)
SPAN=$(( LAST - FIRST + 1 ))
[ "$SPAN" -gt 40 ] && SPAN=40
echo
echo "=== STEP 3: re-fit events [$FIRST, $((FIRST+SPAN))) serially ==="
cfg="./lf_settle_serial.py"
python3 - "$LAUNCHER" "$cfg" "$FIRST" "$SPAN" "$OUT" <<'PYEOF'
import re, sys, pathlib
src, dst, start, span, out = sys.argv[1:6]
t = pathlib.Path(src).read_text()
for pat, rep in [
    (r"^NPROC\s*=.*$", "NPROC = 1"),
    (r"^EVENT_START_INDEX\s*=.*$", "EVENT_START_INDEX = %s" % start),
    (r"^N_EVENTS\s*=.*$", "N_EVENTS = %s" % span),
    (r"^OUTPUT_FILE\s*=.*$", 'OUTPUT_FILE = "%s/serial.dict"' % out),
    (r"^MAX_INTERNAL_THREADS_PER_WORKER\s*=.*$",
     "MAX_INTERNAL_THREADS_PER_WORKER = 1"),
]:
    t, k = re.subn(pat, rep, t, count=1, flags=re.M)
    if k != 1:
        sys.exit("could not rewrite %r" % pat)
pathlib.Path(dst).write_text(t)
PYEOF
[ $? -eq 0 ] || { echo "launcher rewrite failed"; exit 2; }
python3 "$cfg" > "$OUT/serial.log" 2>&1
src_rc=$?
tail -4 "$OUT/serial.log"
rm -f "$cfg"

echo
echo "================= VERDICT ================="
if [ $src_rc -eq 0 ]; then
  echo "Those events fit FINE serially."
  echo "-> The events are not the problem; workers are being killed under"
  echo "   parallelism. See STEP 2 for whether it was the OOM killer."
  echo "-> Immediate workaround: use a smaller NPROC, or HTCondor with"
  echo "   request_memory/request_cpus instead of a shared interactive node."
else
  echo "Serial re-fit ALSO failed (rc=$src_rc) - an event-dependent bug."
  echo "-> Traceback below; send this to whoever maintains the package."
  tail -40 "$OUT/serial.log"
fi
echo "Logs: $OUT/repro.log  $OUT/serial.log"
