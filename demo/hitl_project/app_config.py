from __future__ import annotations

from pathlib import Path


def feature_flag_enabled(env_path: str = ".env") -> bool:
    content = Path(env_path).read_text(encoding="utf-8")
    values = {}
    for line in content.splitlines():
        if not line.strip() or line.strip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values.get("FEATURE_FLAG") == "on"
