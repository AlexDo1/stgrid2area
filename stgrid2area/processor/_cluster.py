"""
Helpers to correctly size a local Dask cluster on shared/HPC nodes.

Both ``os.cpu_count()`` and dask/distributed's own internal "auto" CPU and
memory detection can silently overcount the CPUs actually allocated to this
job on shared or hyperthreaded HPC nodes (e.g. SLURM binding whole physical
cores even when only one hardware thread per core was requested). The
functions here are stgrid2area's own, SLURM-aware replacements, used instead
of relying on distributed's defaults.
"""

import os
from typing import Union

from distributed.system import MEMORY_LIMIT as _TOTAL_SYSTEM_MEMORY


def _detect_available_cpus() -> int:
    """
    Number of CPUs actually available to this process.

    ``os.cpu_count()`` reports the node's total logical CPU count and ignores
    SLURM/cgroup CPU affinity restrictions (e.g. ``--ntasks-per-node``). On a
    shared or hyperthreaded HPC node this can be 2x+ too high, causing
    ``LocalDaskProcessor`` to start far more workers than the job's memory
    allocation was sized for (each worker getting a fraction of the memory it
    needs) and triggering worker restarts/OOM.

    Under SLURM, prefer SLURM's own task/CPU accounting over
    ``os.sched_getaffinity``: on hyperthreaded nodes SLURM commonly binds
    whole physical cores (both hardware threads) to the job's cgroup even
    when only one thread per core was requested (``--cpus-per-task=1``), so
    the CPU affinity mask can still report 2x the number of workers the job's
    memory allocation (``--mem``) was actually sized for. Falls back to
    ``os.sched_getaffinity`` (respects cgroup restrictions outside SLURM) and
    finally ``os.cpu_count()``.
    """
    if os.environ.get("SLURM_JOB_ID"):
        for env_var in (
            "SLURM_NTASKS_PER_NODE",
            "SLURM_NTASKS",
            "SLURM_CPUS_ON_NODE",
            "SLURM_JOB_CPUS_PER_NODE",
        ):
            value = os.environ.get(env_var)
            if value:
                try:
                    return int(value.split(",")[0].split("(")[0])
                except ValueError:
                    continue

    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        pass  # not available on this platform (e.g. macOS)

    return os.cpu_count() or 1


def _resolve_memory_limit(
    memory_limit: Union[str, int, float, None], n_workers: int
) -> Union[str, int, float, None]:
    """
    Resolve the per-worker ``memory_limit`` passed to ``LocalCluster``.

    distributed's own ``memory_limit="auto"`` divides total system memory by
    ``dask.system.CPU_COUNT`` (host/cgroup/affinity CPU count) -- *not* by
    the actual ``n_workers`` requested here. On HPC nodes where CPU affinity
    reports more logical CPUs than intended (e.g. SLURM binding whole
    physical cores, exposing both hyperthread siblings, even though only
    ``--ntasks-per-node=60`` was requested), this silently starves every
    worker of memory regardless of ``n_workers``: observed in production,
    the logged "Worker memory limit" stayed ~2 GiB whether ``n_workers`` was
    120 or 60, because distributed always divided total memory by the
    (wrong) 120 auto-detected cores. Explicitly dividing total system memory
    by our own correctly-detected ``n_workers`` avoids this.
    """
    if memory_limit != "auto" or n_workers <= 0:
        return memory_limit
    return int(_TOTAL_SYSTEM_MEMORY / n_workers)
