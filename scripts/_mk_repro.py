"""Write a launcher copy with a short no-result timeout."""
import re, sys, pathlib
src, dst, timeout = sys.argv[1:4]
t = pathlib.Path(src).read_text()
t, k = re.subn(r"^EVENT_RESULT_STALL_TIMEOUT_SECONDS\s*=.*$",
               "EVENT_RESULT_STALL_TIMEOUT_SECONDS = %s" % timeout,
               t, count=1, flags=re.M)
if k != 1:
    sys.exit("could not set EVENT_RESULT_STALL_TIMEOUT_SECONDS in " + src)
outer = int(timeout) + 60
t, n = re.subn(r"^CHILD_STALL_TIMEOUT_SECONDS\s*=.*$",
               "CHILD_STALL_TIMEOUT_SECONDS = %d" % outer, t, count=1, flags=re.M)
if n:
    print("  outer chunk watchdog set to %d s (inner timeout %s s)" % (outer, timeout))
pathlib.Path(dst).write_text(t)
