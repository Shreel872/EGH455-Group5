"""Store complete product snapshots as JSON Lines."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import threading
from typing import Any
from uuid import uuid4


LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class Event:
    event_type: str
    value: Any
    source: str

    observed_at: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )

    metadata: dict[str, Any] = field(default_factory=dict)

    event_id: str = field(
        default_factory=lambda: uuid4().hex
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventStorage:
    """Append and read events using one shared storage instance."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, event: Event) -> None:
        # Serialize before opening the file. Reject NaN and Infinity.
        encoded = (
            json.dumps(
                event.to_dict(),
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")

        with self._lock:
            self.path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            # Do not attach a new record to a damaged final line.
            if self.path.exists() and self.path.stat().st_size:
                with self.path.open("rb") as handle:
                    handle.seek(-1, 2)

                    if handle.read(1) != b"\n":
                        raise RuntimeError(
                            f"{self.path} has an unfinished final record. "
                            "Recover or remove that partial record "
                            "before appending."
                        )

            with self.path.open("ab") as handle:
                handle.write(encoded)

    def read_all(self) -> list[dict[str, Any]]:
        with self._lock:
            if not self.path.exists():
                return []

            contents = self.path.read_bytes()

        records = []
        lines = contents.split(b"\n")

        for index, line in enumerate(lines):
            if not line.strip():
                continue

            try:
                records.append(json.loads(line))

            except (json.JSONDecodeError, UnicodeDecodeError):
                is_unfinished_tail = (
                    index == len(lines) - 1
                    and not contents.endswith(b"\n")
                )

                if is_unfinished_tail:
                    LOG.warning(
                        "Ignoring incomplete final JSONL record in %s",
                        self.path,
                    )
                    break

                # Corruption in a completed record should be visible.
                raise

        return records

    def pending(self) -> list[dict[str, Any]]:
        """Compatibility with your current uploader.

        Currently returns all records, not just unuploaded records.
        """
        return self.read_all()