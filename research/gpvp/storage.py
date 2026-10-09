"""Append-only JSONL/SQLite storage and optional Parquet export."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol, Self


class RecordWriter(Protocol):
    def write(self, record: dict[str, Any]) -> None: ...

    def close(self) -> None: ...


class JsonlWriter:
    """Write one self-contained provenance record per line; flush each record."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8", newline="\n")
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        encoded = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self._lock:
            self._handle.write(encoded + "\n")
            self._handle.flush()
            os.fsync(self._handle.fileno())

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class SQLiteWriter:
    """Store canonical JSON records in SQLite with queryable provenance columns."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS records (
                record_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                timestamp_utc TEXT NOT NULL,
                provider TEXT,
                model_id TEXT,
                condition TEXT,
                tier TEXT,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_records_run ON records(run_id)"
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_records_condition ON records(condition)"
        )
        self._connection.commit()
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        payload = json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO records (
                    record_id, run_id, timestamp_utc, provider, model_id, condition, tier, status, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(record["record_id"]),
                    str(record["run_id"]),
                    str(record["timestamp_utc"]),
                    record.get("provider"),
                    record.get("model_id"),
                    record.get("condition"),
                    record.get("tier"),
                    str(record.get("status", "unknown")),
                    payload,
                ),
            )
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def open_writer(path: str | Path, format_name: str | None = None) -> RecordWriter:
    output = Path(path)
    fmt = (format_name or output.suffix.lstrip(".")).casefold()
    if fmt in {"jsonl", "ndjson"}:
        return JsonlWriter(output)
    if fmt in {"sqlite", "db", "sqlite3"}:
        return SQLiteWriter(output)
    raise ValueError(
        "Output format must be JSONL or SQLite. Use export-parquet for Parquet."
    )


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    suffix = source.suffix.casefold()
    if suffix in {".jsonl", ".ndjson"}:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} of {source}."
                    ) from exc
                if not isinstance(value, dict):
                    raise TypeError(
                        f"Record on line {line_number} of {source} is not a JSON object."
                    )
                yield value
        return
    if suffix in {".sqlite", ".db", ".sqlite3"}:
        connection = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
        try:
            for (payload,) in connection.execute(
                "SELECT payload_json FROM records ORDER BY rowid"
            ):
                value = json.loads(payload)
                if isinstance(value, dict):
                    yield value
        finally:
            connection.close()
        return
    if suffix == ".parquet":
        try:
            from pyarrow import parquet
        except ImportError as exc:
            raise RuntimeError(
                "Parquet input requires optional dependency pyarrow."
            ) from exc
        table = parquet.read_table(source)
        for row in table.to_pylist():
            payload = row.get("payload_json")
            if isinstance(payload, str):
                value = json.loads(payload)
            else:
                value = row
            if isinstance(value, dict):
                yield value
        return
    raise ValueError(f"Unsupported dataset format: {source.suffix or '<none>'}")


def write_logits_sidecar(
    output_path: str | Path,
    record_id: str,
    token_ids: list[int],
    logits: list[list[float]],
) -> dict[str, str]:
    """Persist optional full-vocabulary logits as a compressed, hashed sidecar."""
    output = Path(output_path)
    sidecar_dir = output.parent / f"{output.name}.logits"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    final_path = sidecar_dir / f"{record_id}.json.gz"
    payload = json.dumps(
        {"record_id": record_id, "token_ids": token_ids, "logits": logits},
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    temporary_path = final_path.with_suffix(final_path.suffix + ".tmp")
    with temporary_path.open("wb") as raw_file:
        with gzip.GzipFile(fileobj=raw_file, mode="wb", mtime=0) as compressed:
            compressed.write(payload)
        raw_file.flush()
        os.fsync(raw_file.fileno())
    os.replace(temporary_path, final_path)
    digest = hashlib.sha256(final_path.read_bytes()).hexdigest()
    return {
        "path": str(final_path.relative_to(output.parent)),
        "sha256": digest,
        "encoding": "gzip-json",
    }


def export_parquet(source: str | Path, destination: str | Path) -> int:
    """Convert JSONL/SQLite records to a Parquet file (optional pyarrow)."""
    try:
        import pyarrow as pa
        from pyarrow import parquet
    except ImportError as exc:
        raise RuntimeError(
            "Parquet export requires optional dependency pyarrow (pip install pyarrow)."
        ) from exc

    records = list(iter_records(source))
    rows = [
        {
            "record_id": str(record.get("record_id", "")),
            "run_id": str(record.get("run_id", "")),
            "timestamp_utc": str(record.get("timestamp_utc", "")),
            "provider": record.get("provider"),
            "model_id": record.get("model_id"),
            "condition": record.get("condition"),
            "tier": record.get("tier"),
            "status": record.get("status"),
            "payload_json": json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        }
        for record in records
    ]
    table = pa.Table.from_pylist(rows)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(table, target, compression="zstd")
    return len(rows)
