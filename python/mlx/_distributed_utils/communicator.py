"""JACCL communicator for distributed MLX operations.

Wraps mx.distributed collective operations in a vLLM-compatible interface.
JACCL handles the RDMA transport over Thunderbolt 5 — this module provides
the higher-level coordination primitives needed by the distributed worker.

Key design: MLX tensor parallelism uses implicit all_reduce inside sharded
layers (AllToShardedLinear / ShardedToAllLinear). The communicator is used
for explicit coordination (parameter broadcast, signal passing) while the
model's sharded layers handle data-plane communication internally.
"""

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)

# Signal constants for rank coordination (matches distributed_server.py)
SIGNAL_GENERATE = 1
SIGNAL_SHUTDOWN = 0


class JACCLCommunicator:
    """Wraps mx.distributed for inter-rank communication via JACCL.

    Provides both control-plane operations (signal passing, parameter
    broadcast) and data-plane operations (all_sum, all_reduce) over
    Thunderbolt 5 RDMA.
    """

    def __init__(self, world):
        """Initialize with an mx.distributed group.

        Args:
            world: Result of mx.distributed.init(backend="jaccl").
        """
        self.world = world
        self._rank = world.rank()
        self._size = world.size()

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._size

    @property
    def is_primary(self) -> bool:
        return self._rank == 0

    def barrier(self):
        """Synchronize all ranks via all_sum of zeros."""
        sync = mx.zeros((1,), dtype=mx.int32)
        sync = mx.distributed.all_sum(sync, group=self.world)
        mx.eval(sync)

    def broadcast_signal(self, signal: int = SIGNAL_GENERATE):
        """Broadcast a signal from rank 0 to all ranks.

        Rank 0 sets the signal value; all other ranks set 0.
        After all_sum, every rank sees the signal.
        """
        if self.is_primary:
            sig = mx.array([signal], dtype=mx.int32)
        else:
            sig = mx.zeros((1,), dtype=mx.int32)
        sig = mx.distributed.all_sum(sig, group=self.world)
        mx.eval(sig)
        return int(sig.item())

    def broadcast_generation_params(
        self,
        prompt_tokens: list[int] | None = None,
        max_tokens: int = 0,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> tuple:
        """Broadcast generation parameters from rank 0 to all ranks.

        Rank 0 provides real values; workers provide zeros. all_sum
        distributes them to all ranks.

        Returns:
            (prompt_array, max_tokens, temperature, top_p) — all ranks
            receive the same values.
        """
        if self.is_primary:
            if prompt_tokens is None:
                raise ValueError("Rank 0 must provide prompt_tokens")
            params = mx.array([max_tokens, len(prompt_tokens)], dtype=mx.int32)
            fparams = mx.array([temperature, top_p], dtype=mx.float32)
        else:
            params = mx.zeros((2,), dtype=mx.int32)
            fparams = mx.zeros((2,), dtype=mx.float32)

        params = mx.distributed.all_sum(params, group=self.world)
        fparams = mx.distributed.all_sum(fparams, group=self.world)
        mx.eval(params, fparams)

        n_max = int(params[0].item())
        n_prompt = int(params[1].item())
        temp = float(fparams[0].item())
        tp = float(fparams[1].item())

        # Broadcast prompt tokens
        if self.is_primary:
            tok_buf = mx.array(prompt_tokens, dtype=mx.int32)
        else:
            tok_buf = mx.zeros((n_prompt,), dtype=mx.int32)
        tok_buf = mx.distributed.all_sum(tok_buf, group=self.world)
        mx.eval(tok_buf)

        return tok_buf, n_max, temp, tp

    def all_sum(self, tensor):
        """All-reduce sum across all ranks."""
        result = mx.distributed.all_sum(tensor, group=self.world)
        mx.eval(result)
        return result
