"""Validate the supplied files and an actual Parquet write/read roundtrip.

Run from the project root after installing requirements.txt:
    python scripts/validate_real_parquet.py
No data or network request leaves the computer.
"""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pyarrow as pa
import pyarrow.parquet as pq
from moneygraph.data import load


def main():
    started = time.perf_counter()
    source = ROOT / "data"
    original = load(source, "parquet")
    tables = {}
    checks = {}
    with tempfile.TemporaryDirectory(prefix="moneygraph-roundtrip-") as folder:
        target = Path(folder)
        for name in ("nodes", "edges", "transactions"):
            path = source / f"{name}.parquet"
            table = pq.read_table(path)
            tables[name] = table
            output = target / path.name
            pq.write_table(table, output)
            restored_table = pq.read_table(output)
            assert table.equals(restored_table, check_metadata=True), name
            checks[name] = {
                "rows": table.num_rows,
                "schema": str(table.schema),
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "roundtrip_equal": True,
            }
        restored = load(target, "parquet")
        assert original.nodes == restored.nodes
        assert original.edges == restored.edges
        assert original.transactions == restored.transactions

    result = {
        "status": "passed",
        "source": "Three supplied organizer Parquet files in data/",
        "pyarrow": pa.__version__,
        "tables": checks,
        "loader_roundtrip_equal": True,
        "edges_reconcile_transactions": True,
        "nodes": len(original.nodes),
        "edges": len(original.edges),
        "transactions": len(original.transactions),
        "seed_clients": sum(n["is_seed"] for n in original.nodes.values()),
        "total_transaction_turnover_kzt": sum(t["sum_kzt"] for t in original.transactions),
        "first_date": min(t["date"] for t in original.transactions),
        "last_date": max(t["date"] for t in original.transactions),
        "warnings": original.warnings,
        "runtime_seconds": round(time.perf_counter() - started, 4),
        "scope": "Integrity and reproducibility; not validation of suspected roles or criminal activity.",
    }
    destination = ROOT / "results" / "checks" / "real-parquet-roundtrip.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "tables"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
