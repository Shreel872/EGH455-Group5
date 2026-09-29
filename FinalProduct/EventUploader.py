"""Upload JSONL events without repeatedly sending acknowledged records."""

import json
import logging
import os
from pathlib import Path
import threading
from urllib.error import URLError
from urllib.request import Request, urlopen


LOG = logging.getLogger(__name__)


class EventUploader:
    def __init__(
        self,
        url=None,
        ack_path="data/uploaded.json",
        token=None,
        timeout=5,
    ):
        self.url = url
        self.ack_path = Path(ack_path)
        self.token = token
        self.timeout = timeout

        self.lock = threading.Lock()
        self.acknowledged = set()

        if self.ack_path.exists():
            ids = json.loads(
                self.ack_path.read_text(encoding="utf-8")
            )

            if (
                not isinstance(ids, list)
                or not all(isinstance(value, str) for value in ids)
            ):
                raise ValueError(
                    "Invalid upload acknowledgement file"
                )

            self.acknowledged = set(ids)

    def upload(self, records, stop=None):
        """Return the number of newly acknowledged events."""
        if not self.url:
            return 0

        sent = 0

        with self.lock:
            for record in records:
                if stop is not None and stop.is_set():
                    break

                event_id = record.get("event_id")

                if not isinstance(event_id, str) or not event_id:
                    raise ValueError(
                        "An old event has no event_id. "
                        "Migrate that log or use a new log "
                        "before uploading."
                    )

                if event_id in self.acknowledged:
                    continue

                headers = {
                    "Content-Type": "application/json",
                    "Idempotency-Key": event_id,
                }

                if self.token:
                    headers["Authorization"] = (
                        f"Bearer {self.token}"
                    )

                request = Request(
                    self.url,
                    data=json.dumps(
                        record,
                        allow_nan=False,
                    ).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )

                try:
                    with urlopen(
                        request,
                        timeout=self.timeout,
                    ) as response:
                        if not 200 <= response.status < 300:
                            LOG.warning(
                                "Upload not acknowledged: HTTP %s",
                                response.status,
                            )
                            break

                except (URLError, OSError) as exc:
                    LOG.warning(
                        "Event upload failed; will retry: %s",
                        exc,
                    )
                    break

                acknowledged = (
                    self.acknowledged | {event_id}
                )

                self.ack_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                temporary = self.ack_path.with_suffix(
                    self.ack_path.suffix + ".tmp"
                )

                with temporary.open(
                    "w",
                    encoding="utf-8",
                ) as handle:
                    json.dump(
                        sorted(acknowledged),
                        handle,
                    )

                    handle.flush()
                    os.fsync(handle.fileno())

                os.replace(
                    temporary,
                    self.ack_path,
                )

                self.acknowledged = acknowledged
                sent += 1

        return sent