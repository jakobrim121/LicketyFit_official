# Bundled runtime fallback

This release package includes `threadpoolctl` 3.6.0 as a pure-Python fallback.
The launchers put this directory first on their own and their child processes'
Python paths, so an existing environment that predates the dependency can still
run the fitter. Source checkouts should continue to install `requirements.txt`.

`threadpoolctl.py` is copied unchanged from the upstream 3.6.0 distribution:

- Project: <https://github.com/joblib/threadpoolctl>
- License: BSD 3-Clause; see `LICENSE.threadpoolctl.txt`

This fallback changes dependency bootstrapping only. It does not change the
likelihood, physics model, optimizer, event selection, or fitted estimates.
