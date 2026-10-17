"""Lightweight MOV container checks performed before a job is admitted."""

from __future__ import annotations

from pathlib import Path


def has_valid_mov_container(path: Path) -> bool:
    """Check top-level ISO-BMFF atom boundaries and required MOV atoms.

    This detects truncated or malformed containers. It does not decode video frames;
    the configured analyzer remains responsible for validating usable visual content.
    """
    file_size = path.stat().st_size
    offset = 0
    seen_atoms: set[bytes] = set()
    atom_count = 0

    with path.open("rb") as video:
        while offset < file_size:
            if file_size - offset < 8 or atom_count >= 100_000:
                return False
            video.seek(offset)
            header = video.read(8)
            if len(header) != 8:
                return False
            atom_size = int.from_bytes(header[:4], "big")
            atom_type = header[4:8]
            header_size = 8

            if atom_size == 1:
                extended_size = video.read(8)
                if len(extended_size) != 8:
                    return False
                atom_size = int.from_bytes(extended_size, "big")
                header_size = 16
            elif atom_size == 0:
                atom_size = file_size - offset

            if atom_size < header_size or offset + atom_size > file_size:
                return False

            seen_atoms.add(atom_type)
            atom_count += 1
            offset += atom_size

    return {b"ftyp", b"moov", b"mdat"}.issubset(seen_atoms)
