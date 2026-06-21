from __future__ import annotations

from enum import StrEnum


class Visibility(StrEnum):
    READ_ONLY = "read_only"
    WRITABLE = "writable"
    HIDDEN = "hidden"

