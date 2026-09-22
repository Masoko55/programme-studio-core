"""One process-safe lock shared by both Python services on the single host."""
import asyncio
import fcntl
import time
from pathlib import Path
from app.config.settings import settings

class GPULease:
    def __init__(self, lease_path: Path | None = None):
        self.lease_path = lease_path or settings.gpu_lease_path
        self.file = None

    def acquire(self) -> None:
        self.lease_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.lease_path.open("a+")
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.file.close()
            self.file = None
            raise RuntimeError("GPU lease is already held") from error

    def release(self) -> None:
        if self.file is not None:
            fcntl.flock(self.file, fcntl.LOCK_UN)
            self.file.close()
            self.file = None

    async def __aenter__(self):
        deadline = time.monotonic() + settings.gpu_lease_timeout_seconds
        while True:
            try:
                self.acquire()
                return self
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                await asyncio.sleep(1)

    async def __aexit__(self, *args):
        self.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()
