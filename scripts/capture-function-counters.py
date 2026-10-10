#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read-only Spark function-specific RDMA counter capture; run on a Spark."""
import datetime
import os
import json
from pathlib import Path
import re
import subprocess
import time

HCAS = json.loads(os.environ["SPARK_RING_REQUIRED_HCAS"])
if len(HCAS) != 4 or len(set(HCAS)) != 4 or any(not re.fullmatch(r"[A-Za-z0-9_.:-]+", h) for h in HCAS):
    raise SystemExit("Exactly four explicit HCA function names required")
REQUIRED = ("rx_vport_rdma_unicast_bytes", "tx_vport_rdma_unicast_bytes", "rx_vport_rdma_unicast_packets", "tx_vport_rdma_unicast_packets")
result = {"schema": "spark-ring-function-counters/v1", "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "monotonic_ns": time.monotonic_ns(), "devices": {}}
for hca in HCAS:
    d = Path("/sys/class/infiniband") / hca
    ports = d / "ports/1"
    names = sorted(x.name for x in (d / "device/net").iterdir())
    if len(names) != 1:
        raise SystemExit(f"{hca}: exactly one associated netdev required, got {names!r}")
    netdev = names[0]
    r = subprocess.run(["/usr/sbin/ethtool", "-S", netdev], text=True, capture_output=True, check=True)
    stats = {m.group(1): int(m.group(2)) for line in r.stdout.splitlines() if (m := re.fullmatch(r"\s*([A-Za-z0-9_]+):\s*([0-9]+)\s*", line))}
    missing = [name for name in REQUIRED if name not in stats]
    if missing:
        raise SystemExit(f"{hca}/{netdev}: mandatory function-specific counters missing: {missing}")
    hw = {}
    for field in ("rx_write_requests", "rx_read_requests", "rx_atomic_requests", "out_of_buffer", "req_cqe_error", "resp_cqe_error", "req_transport_retries_exceeded", "req_rnr_retries_exceeded", "packet_seq_err", "local_ack_timeout_err"):
        f = ports / "hw_counters" / field
        if f.is_file():
            hw[field] = int(f.read_text().strip())
    mandatory_errors = {"out_of_buffer", "req_cqe_error", "resp_cqe_error", "req_transport_retries_exceeded", "req_rnr_retries_exceeded", "packet_seq_err", "local_ack_timeout_err"}
    if not mandatory_errors.issubset(hw):
        raise SystemExit(f"{hca}: qualified transport error counters are unavailable")
    physical = {}
    for field in ("port_xmit_data", "port_rcv_data", "port_xmit_packets", "port_rcv_packets"):
        f = ports / "counters" / field
        if f.is_file():
            physical[field] = int(f.read_text().strip())
    result["devices"][hca] = {"netdev": netdev, "pci_address": (d / "device").resolve().name, "function_rdma": {name: stats[name] for name in REQUIRED}, "ib_port_counters_crosscheck": physical, "hw_requests_and_errors": hw}
try:
    r = subprocess.run(["rdma", "-j", "resource", "show", "qp"], text=True, capture_output=True)
    result["live_qps"] = json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else []
    result["live_qps_capture_returncode"] = r.returncode
except FileNotFoundError:
    result["live_qps"] = []
    result["live_qps_capture_returncode"] = 127
print(json.dumps(result, indent=2))
