"""Local human review notes, independent of model tools and banking actions."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import threading


DECISIONS = frozenset(("investigate", "request_info", "no_action"))


class ReviewError(Exception):
    def __init__(self, message, code="storage_error", status=500):
        super().__init__(message)
        self.code = code
        self.status = status


def _valid_hashes(value):
    return (isinstance(value, dict) and bool(value)
            and all(isinstance(key, str) and key and isinstance(item, str) and item
                    for key, item in value.items()))


class ReviewStore:
    """Append-only JSONL journal; the last event per gid is its current review.

    Matching input file hashes separate datasets even when they reuse gids.
    A lock per resolved file protects all instances within this server process.
    The journal does not provide multi-process locking or user authentication.
    """
    _locks = {}
    _locks_guard = threading.Lock()

    def __init__(self, path, dataset_hashes, known_gids):
        if not _valid_hashes(dataset_hashes):
            raise ReviewError("Для журнала нужны непустые контрольные суммы выгрузки.", "invalid_dataset", 400)
        self.path = Path(path).resolve()
        self.dataset_hashes = dict(dataset_hashes)
        self.known_gids = frozenset(str(gid) for gid in known_gids)
        with self._locks_guard:
            self._lock = self._locks.setdefault(os.path.normcase(str(self.path)), threading.RLock())

    def _read(self):
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return [], False
        except (OSError, UnicodeError) as error:
            raise ReviewError("Не удалось прочитать локальный журнал решений.") from error
        events = []
        for line_number, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                if (not isinstance(event, dict)
                        or not isinstance(event.get("timestamp"), str)
                        or not isinstance(event.get("gid"), str)
                        or len(event["gid"]) > 20
                        or not re.fullmatch(r"-?(0|[1-9][0-9]*)", event["gid"])
                        or not -(2**63) <= int(event["gid"]) < 2**63
                        or event.get("decision") not in DECISIONS
                        or not isinstance(event.get("note"), str)
                        or len(event["note"]) > 1000
                        or not _valid_hashes(event.get("input_sha256"))):
                    raise ValueError("Invalid journal event")
                timestamp = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
                if timestamp.tzinfo is None:
                    raise ValueError("Timestamp without timezone")
            except (ValueError, TypeError) as error:
                raise ReviewError(f"Журнал повреждён в строке {line_number}; существующие записи сохранены. Восстановите журнал из копии.", "invalid_store") from error
            events.append(event)
        return events, bool(raw and not raw.endswith(("\n", "\r")))

    def list(self):
        with self._lock:
            events, _ = self._read()
        current = [event for event in events if event["input_sha256"] == self.dataset_hashes
                   and event["gid"] in self.known_gids]
        latest = {event["gid"]: event for event in current}
        return {"reviews": [latest[gid] for gid in sorted(latest, key=int)],
                "events_count": len(current)}

    def record(self, gid, decision, note=""):
        if (not isinstance(gid, str) or not re.fullmatch(r"-?(0|[1-9][0-9]*)", gid)
                or len(gid) > 20 or not -(2**63) <= int(gid) < 2**63):
            raise ReviewError("gid должен быть точной строкой идентификатора int64.", "invalid_gid", 400)
        if gid not in self.known_gids:
            raise ReviewError("Участник отсутствует в текущей выгрузке.", "unknown_gid", 404)
        if not isinstance(decision, str) or decision not in DECISIONS:
            raise ReviewError("Выберите investigate, request_info или no_action.", "invalid_decision", 400)
        if not isinstance(note, str) or len(note) > 1000:
            raise ReviewError("Комментарий должен быть строкой длиной не более 1000 символов.", "invalid_note", 400)
        with self._lock:
            _, needs_separator = self._read()  # Never append to a corrupt tail.
            event = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                     "gid": gid, "decision": decision, "note": note,
                     "input_sha256": dict(self.dataset_hashes)}
            line = json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n"
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                    if needs_separator:
                        handle.write("\n")
                    handle.write(line)
                    handle.flush()
                    os.fsync(handle.fileno())
            except (OSError, UnicodeError) as error:
                raise ReviewError("Не удалось сохранить решение в локальном журнале. Проверьте доступ к папке и свободное место.") from error
            return event
