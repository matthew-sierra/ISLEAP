"""Fingerprints and atomic file writes.

Atomic write: write `name.partial`, close, fsync, then os.replace() it to
`name`. A rename on one filesystem either fully happens or not at all, so a
file under its final name is always complete; a crashed run leaves only a
`.partial`, which nothing downstream reads.
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import h5py

FINGERPRINT_BYTES = 4 * 1024 * 1024


def fingerprint(path: Path) -> dict:
    """Cheap identity for a large input: size + SHA-256 of its first 4 MB."""
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(FINGERPRINT_BYTES))
    return {"path": str(path.resolve()), "size": path.stat().st_size,
            "sha256_head": h.hexdigest()}


def digest(obj) -> str:
    """Stable short hash of any JSON-serialisable object."""
    s = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(s).hexdigest()[:16]


def _fsync(path: Path) -> None:
    flags = os.O_RDWR if os.name == "nt" else os.O_RDONLY  # Windows needs write access
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def atomic_h5(path: Path):
    """Open `path.partial` for writing; on success close, fsync, rename to `path`."""
    path = Path(path)
    tmp = path.with_name(path.name + ".partial")
    tmp.unlink(missing_ok=True)
    f = h5py.File(tmp, "w")
    try:
        yield f
    except BaseException:
        f.close()
        raise  # leave the .partial for inspection; it is never read
    f.close()  # must be closed before renaming (Windows)
    _fsync(tmp)
    os.replace(tmp, path)
