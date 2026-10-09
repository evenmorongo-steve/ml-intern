from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from research.gpvp.storage import (
    JsonlWriter,
    SQLiteWriter,
    iter_records,
    write_logits_sidecar,
)


class StorageTests(unittest.TestCase):
    def test_jsonl_and_sqlite_round_trip(self) -> None:
        record = {
            "record_id": "rec-1",
            "run_id": "run-1",
            "timestamp_utc": "2026-10-08T00:00:00Z",
            "provider": "fixture",
            "model_id": "fixture-model",
            "condition": "test",
            "tier": "UNPRIMED",
            "status": "completed",
            "user_prompt": "test prompt",
        }
        with tempfile.TemporaryDirectory() as directory:
            jsonl_path = Path(directory) / "rows.jsonl"
            with JsonlWriter(jsonl_path) as writer:
                writer.write(record)
            self.assertEqual(list(iter_records(jsonl_path)), [record])

            sqlite_path = Path(directory) / "rows.sqlite"
            with SQLiteWriter(sqlite_path) as writer:
                writer.write(record)
            self.assertEqual(list(iter_records(sqlite_path)), [record])

    def test_full_logits_sidecar_is_hashed_and_compressed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run.jsonl"
            manifest = write_logits_sidecar(
                output, "record-1", [1, 2], [[0.1, 0.2], [0.3, 0.4]]
            )
            path = Path(directory) / manifest["path"]
            self.assertTrue(path.exists())
            self.assertEqual(len(manifest["sha256"]), 64)
            self.assertEqual(manifest["encoding"], "gzip-json")


if __name__ == "__main__":
    unittest.main()
