"""JACCL initialization for Ray workers.

Replaces mlx.launch by manually setting the environment variables that
MLX's JACCL backend expects, then calling mx.distributed.init().

This allows Ray to manage MLX distributed workers without needing the
mlx.launch binary.

Required env vars (set before mx.distributed.init):
    MLX_RANK              - integer rank of this worker
    MLX_JACCL_COORDINATOR - "ip:port" of the rank-0 coordinator
    MLX_IBV_DEVICES       - path to JSON file with RDMA device mapping

The IBV devices file contains the full N×N RDMA topology:
    [[null, "rdma_en5"], ["rdma_en5", null]]
"""

import json
import logging
import os
import tempfile
import time

logger = logging.getLogger(__name__)


def init_jaccl_in_ray_worker(
    rank: int,
    world_size: int,
    coordinator_addr: str,
    ibv_devices_json: str,
    max_retries: int = 3,
    retry_delay: float = 2.0,
):
    """Initialize JACCL distributed group inside a Ray worker.

    Args:
        rank: This worker's rank (0-indexed).
        world_size: Total number of workers.
        coordinator_addr: "ip:port" of the rank-0 coordinator.
        ibv_devices_json: JSON string of the RDMA device topology
            (same format as mlx.launch writes to MLX_IBV_DEVICES).
        max_retries: Number of init retries on failure.
        retry_delay: Seconds between retries (exponential backoff).

    Returns:
        The mx.distributed group object.

    Raises:
        RuntimeError: If JACCL initialization fails after all retries.
    """
    import mlx.core as mx

    # Write IBV devices to temp file (JACCL reads from file path, not env value)
    fd, ibv_path = tempfile.mkstemp(prefix=f"mlx_ibv_rank{rank}_", suffix=".json")
    with os.fdopen(fd, "w") as f:
        f.write(ibv_devices_json)

    # Set environment variables that JACCL reads during init
    os.environ["MLX_RANK"] = str(rank)
    os.environ["MLX_JACCL_COORDINATOR"] = coordinator_addr
    os.environ["MLX_IBV_DEVICES"] = ibv_path

    logger.info(
        "Rank %d: JACCL env set — coordinator=%s, ibv_devices=%s",
        rank, coordinator_addr, ibv_path,
    )

    last_error = None
    for attempt in range(max_retries):
        try:
            world = mx.distributed.init(backend="jaccl", strict=True)

            if world.size() != world_size:
                raise RuntimeError(
                    f"Expected world_size={world_size}, got {world.size()}"
                )
            if world.rank() != rank:
                raise RuntimeError(
                    f"Expected rank={rank}, got {world.rank()}"
                )

            logger.info(
                "Rank %d/%d: JACCL initialized successfully",
                rank, world_size,
            )
            return world

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                delay = retry_delay * (2 ** attempt)
                logger.warning(
                    "Rank %d: JACCL init attempt %d failed: %s. Retrying in %.1fs...",
                    rank, attempt + 1, e, delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "Rank %d: JACCL init failed after %d attempts: %s",
                    rank, max_retries, e,
                )

    raise RuntimeError(
        f"Rank {rank}: JACCL initialization failed after {max_retries} attempts: {last_error}"
    )


def verify_jaccl_group(world) -> dict:
    """Verify that a JACCL group is functional by running all_sum.

    Returns a dict with rank, world_size, and test result.
    """
    import mlx.core as mx

    # Each rank contributes its rank value; sum should be N*(N-1)/2
    test = mx.array([world.rank()], dtype=mx.int32)
    result = mx.distributed.all_sum(test)
    mx.eval(result)

    expected = world.size() * (world.size() - 1) // 2
    actual = int(result.item())

    return {
        "rank": world.rank(),
        "world_size": world.size(),
        "all_sum_expected": expected,
        "all_sum_actual": actual,
        "ok": actual == expected,
    }
