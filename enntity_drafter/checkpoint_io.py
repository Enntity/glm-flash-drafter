"""Bounded checkpoint writes without importing model or tensor dependencies."""
import errno
import os
import shutil
from pathlib import Path


class WrittenBack:
    """A write-only file for ``torch.save`` that writes back and drops what it wrote every ``chunk`` bytes: a
    checkpoint is many GB of dirty page cache otherwise, which MemAvailable does not count as free until the kernel
    has written it (GB10: the same memory the GPU uses)."""

    chunk = 256 * 2**20

    def __init__(self, path: Path, min_free_bytes: int = 16 * 2**30):
        if min_free_bytes < 0:
            raise ValueError("minimum disk headroom must be nonnegative")
        self.path, self.min_free_bytes = path, min_free_bytes
        self.check_space(0)
        self.stream, self.written, self.synced = open(path, "wb"), 0, 0

    def check_space(self, incoming: int) -> None:
        free = shutil.disk_usage(self.path.parent).free
        if free < self.min_free_bytes + incoming:
            raise OSError(errno.ENOSPC,
                          f"checkpoint stopped: {free} free bytes cannot fit {incoming} bytes "
                          f"and preserve {self.min_free_bytes} bytes of disk headroom", str(self.path))

    def write(self, data) -> int:
        self.check_space(len(data))
        self.stream.write(data)
        self.written += len(data)
        if self.written - self.synced >= self.chunk:
            self.sync()
        return len(data)

    def flush(self) -> None:
        self.stream.flush()

    def sync(self) -> None:
        self.stream.flush()
        os.fsync(self.stream.fileno())
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(self.stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        self.synced = self.written

    def close(self) -> None:
        try:
            self.sync()
        finally:
            self.stream.close()

