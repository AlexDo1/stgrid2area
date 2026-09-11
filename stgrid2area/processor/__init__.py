"""
Parallel processing of areas against spatiotemporal grids using Dask.

Public API
----------
``LocalDaskProcessor``
    Single-machine / single-node processing using a local ``LocalCluster``.
``MPIDaskProcessor``
    Multi-node HPC processing using an externally provided ``dask-mpi``
    cluster.

Both classes share the same core execution loop (``_engine._run_with_client``)
and just differ in how the Dask ``Client`` is created.

Module layout
-------------
``_cluster``
    Helpers to size a local cluster correctly on shared/HPC nodes
    (``_detect_available_cpus``, ``_resolve_memory_limit``).
``_progress``
    Heartbeat/progress logging, including an on-disk ground-truth cross-check.
``_engine``
    ``_run_with_client``: the shared task submission/collection loop.
``local`` / ``mpi``
    The public ``LocalDaskProcessor`` / ``MPIDaskProcessor`` classes.
"""

import rioxarray  # noqa: F401 -- registers the .rio xarray accessor

from .local import LocalDaskProcessor
from .mpi import MPIDaskProcessor

__all__ = ["LocalDaskProcessor", "MPIDaskProcessor"]
