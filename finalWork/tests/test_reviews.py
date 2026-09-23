from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from moneygraph.reviews import ReviewError, ReviewStore


GID = "100000000331309100"
OTHER = "100000000331309101"
HASHES = {"nodes.parquet": "a" * 64, "transactions.parquet": "b" * 64}


class ReviewStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "journal" / "reviews.jsonl"
        self.store = ReviewStore(self.path, HASHES, [GID, OTHER])

    def tearDown(self):
        self.temp.cleanup()

    def test_empty_store_is_read_only(self):
        self.assertEqual(self.store.list(), {"reviews": [], "events_count": 0})
        self.assertFalse(self.path.parent.exists())

    def test_append_preserves_previous_events_and_restart_loads_latest(self):
        event = self.store.record(GID, "investigate", "Проверить связь\nсо вторым участником.")
        first_bytes = self.path.read_bytes()
        later = self.store.record(GID, "request_info", "Нужна полная входящая история.")
        self.assertTrue(self.path.read_bytes().startswith(first_bytes))
        self.assertEqual(len(self.path.read_text(encoding="utf-8").splitlines()), 2)
        restarted = ReviewStore(self.path, HASHES, [int(GID), int(OTHER)])
        self.assertEqual(restarted.list(), {"reviews": [later], "events_count": 2})
        self.assertEqual(event["input_sha256"], HASHES)
        self.assertEqual(event["gid"], GID)
        self.assertEqual(datetime.fromisoformat(event["timestamp"]).utcoffset(), timezone.utc.utcoffset(None))

    def test_datasets_do_not_share_decisions_even_for_same_gid(self):
        event = self.store.record(GID, "investigate", "Первая выгрузка")
        changed_hashes = dict(HASHES, **{"transactions.parquet": "c" * 64})
        another = ReviewStore(self.path, changed_hashes, [GID, OTHER])
        self.assertEqual(another.list(), {"reviews": [], "events_count": 0})
        other_event = another.record(GID, "no_action", "Вторая выгрузка")
        self.assertEqual(self.store.list(), {"reviews": [event], "events_count": 1})
        self.assertEqual(another.list(), {"reviews": [other_event], "events_count": 1})
        self.assertEqual(len(self.path.read_text(encoding="utf-8").splitlines()), 2)

    def test_invalid_values_do_not_change_journal(self):
        self.store.record(GID, "no_action", "Нуждается в последующей оценке")
        before = self.path.read_bytes()
        values = [(int(GID), "investigate", ""), ("999", "investigate", ""),
                  ("01", "investigate", ""), (str(2**63), "investigate", ""),
                  (GID, "guilty", ""), (GID, None, ""), (GID, [], ""),
                  (GID, "request_info", "x" * 1001), (GID, "request_info", None)]
        for gid, decision, note in values:
            with self.subTest(gid=gid, decision=decision):
                with self.assertRaises(ReviewError):
                    self.store.record(gid, decision, note)
                self.assertEqual(self.path.read_bytes(), before)

    def test_exact_neighboring_int64_gids_have_independent_reviews(self):
        self.store.record(OTHER, "request_info", "Second")
        self.store.record(GID, "investigate", "First")
        self.assertEqual([event["gid"] for event in self.store.list()["reviews"]], [GID, OTHER])

    def test_broken_journal_is_not_overwritten_or_appended(self):
        self.store.record(GID, "investigate")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write('{"truncated":')
        before = self.path.read_bytes()
        for action in (self.store.list, lambda: self.store.record(OTHER, "no_action")):
            with self.assertRaises(ReviewError) as found:
                action()
            self.assertEqual(found.exception.code, "invalid_store")
        self.assertEqual(self.path.read_bytes(), before)

    def test_valid_final_line_without_newline_is_retained(self):
        self.store.record(GID, "investigate")
        self.path.write_bytes(self.path.read_bytes().rstrip(b"\n"))
        self.store.record(OTHER, "request_info")
        self.assertEqual(self.store.list()["events_count"], 2)

    def test_storage_failure_is_a_review_error(self):
        self.path.parent.mkdir(parents=True)
        self.path.mkdir()
        for action in (self.store.list, lambda: self.store.record(GID, "investigate")):
            with self.assertRaises(ReviewError) as found:
                action()
            self.assertEqual(found.exception.code, "storage_error")

    def test_threaded_instances_append_complete_events(self):
        another = ReviewStore(self.path, HASHES, [GID, OTHER])
        def write(index):
            store = self.store if index % 2 else another
            return store.record(GID if index % 2 else OTHER, "request_info", str(index))
        with ThreadPoolExecutor(max_workers=6) as pool:
            events = list(pool.map(write, range(24)))
        lines = self.path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 24)
        self.assertEqual({json.loads(line)["note"] for line in lines}, {event["note"] for event in events})
        self.assertEqual(self.store.list()["events_count"], 24)
        self.assertEqual(len(self.store.list()["reviews"]), 2)

    def test_hashes_are_required_and_copied(self):
        for hashes in ({}, None, {"file": None}):
            with self.assertRaises(ReviewError):
                ReviewStore(self.path, hashes, [GID])
        hashes = dict(HASHES)
        store = ReviewStore(self.path, hashes, [GID])
        hashes["nodes.parquet"] = "different"
        self.assertEqual(store.record(GID, "no_action")["input_sha256"], HASHES)


if __name__ == "__main__":
    unittest.main()
