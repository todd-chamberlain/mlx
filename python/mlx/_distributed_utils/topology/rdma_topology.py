"""TB5 RDMA topology discovery and IBV device mapping.

Parses JACCL hostfile format and generates the IBV device lists that
mlx.distributed expects. Can also auto-detect RDMA devices via ibv_devices.
"""

import json
import logging
import os
import subprocess
import tempfile

logger = logging.getLogger(__name__)


def load_hostfile(path: str) -> list[dict]:
    """Load a JACCL hostfile.

    Format: list of dicts with keys:
        ssh:  hostname for SSH access
        ips:  list of reachable IPs (rank 0 must have at least one)
        rdma: list of RDMA device names (null for self, device name for peer)

    Example for 2 nodes:
        [
            {"ssh": "mac-studio-1.local", "ips": ["10.10.0.1"], "rdma": [null, "rdma_en5"]},
            {"ssh": "mac-studio-2.local", "ips": [], "rdma": ["rdma_en5", null]}
        ]
    """
    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) < 2:
        raise ValueError(f"Hostfile must contain at least 2 entries, got {len(data)}")

    for i, entry in enumerate(data):
        if "ssh" not in entry or "rdma" not in entry:
            raise ValueError(f"Hostfile entry {i} missing required 'ssh' or 'rdma' key")
        if len(entry["rdma"]) != len(data):
            raise ValueError(
                f"Entry {i} rdma list has {len(entry['rdma'])} items, "
                f"expected {len(data)} (one per node)"
            )
        if entry["rdma"][i] is not None:
            raise ValueError(
                f"Entry {i} rdma[{i}] should be null (self-reference), "
                f"got '{entry['rdma'][i]}'"
            )

    if not data[0].get("ips"):
        raise ValueError("Rank 0 must have at least one IP in 'ips' list")

    return data


def get_coordinator_address(hostfile: list[dict], port: int = 9000) -> str:
    """Get the coordinator address (rank 0 IP:port) from a hostfile."""
    return f"{hostfile[0]['ips'][0]}:{port}"


def get_ibv_devices_json(hostfile: list[dict]) -> str:
    """Get the IBV devices JSON string from a hostfile.

    Returns the same format that mlx.launch writes to the MLX_IBV_DEVICES
    temp file: a JSON list of RDMA device arrays, one per node.
    """
    return json.dumps([entry["rdma"] for entry in hostfile])


def write_ibv_devices_file(hostfile: list[dict]) -> str:
    """Write IBV devices JSON to a temp file and return the path.

    The returned path should be set as MLX_IBV_DEVICES env var.
    The file is NOT auto-deleted — caller is responsible for cleanup
    or it will be cleaned up when the process exits via /tmp.
    """
    content = get_ibv_devices_json(hostfile)
    fd, path = tempfile.mkstemp(prefix="mlx_ibv_", suffix=".json")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    logger.debug("Wrote IBV devices to %s: %s", path, content)
    return path


def detect_rdma_devices() -> list[str]:
    """Auto-detect local RDMA devices using ibv_devices.

    Returns a list of device names like ['rdma_en5', 'rdma_en7'].
    Returns empty list if ibv_devices is not available or no devices found.
    """
    try:
        result = subprocess.run(
            ["ibv_devices"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        devices = []
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("rdma_"):
                # ibv_devices output: "    rdma_en5    ..."
                devices.append(stripped.split()[0])
        return devices
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def get_hostnames(hostfile: list[dict]) -> list[str]:
    """Extract SSH hostnames from a hostfile."""
    return [entry["ssh"] for entry in hostfile]


def get_world_size(hostfile: list[dict]) -> int:
    """Get the number of nodes in a hostfile."""
    return len(hostfile)
