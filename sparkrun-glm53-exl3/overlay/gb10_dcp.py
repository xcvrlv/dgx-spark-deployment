"""Allocation-free dispatch for the pinned RoCEnante output-buffer API."""
import os

from vllm.logger import init_logger

logger = init_logger(__name__)
ENABLED = os.getenv("VLLM_GB10_DCP_GATHER_INTO", "0") == "1"


def try_roce_gather_into(group, inp, out, dim=0):
    """Return False only before launching; transport errors remain fail-stop.

    Dispatch uses rank-invariant shape/dtype limits. Pointer alignment is
    handled by the runtime, never by a per-rank transport decision.
    """
    if not ENABLED:
        return False
    comm = getattr(group, "device_communicator", None)
    adapter = getattr(comm, "b12x_ar_comm", None)
    if (not getattr(comm, "use_roce_allreduce", False)
            or getattr(adapter, "backend_name", None) != "B12X_ROCENANTE"
            or getattr(adapter, "disabled", True)):
        return False
    if not adapter.should_all_gather(inp, dim):
        return False
    # R22's adapter discards the existing runtime's out= interface. Calling
    # that pinned runtime retains its lock, stream ordering and health checks.
    adapter._runtime.all_gather(inp, dim=dim, out=out)
    logger.info_once("v16: DCP RoCEnante gather into existing workspace active")
    return True
