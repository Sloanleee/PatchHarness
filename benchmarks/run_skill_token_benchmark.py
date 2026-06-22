from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.context import ContextCompressor  # noqa: E402
from app.skills import SkillManager  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def main() -> None:
    manager = SkillManager.from_default_dir()
    tokenizer = ContextCompressor()
    rows = []
    full_tokens = 0
    frontmatter_tokens = 0

    for frontmatter in manager.list_frontmatter():
        public = frontmatter.to_public_dict()
        content = manager.load_skill(frontmatter.name)
        public_tokens = tokenizer.estimate_tokens(public)
        content_tokens = tokenizer.estimate_tokens(content)
        frontmatter_tokens += public_tokens
        full_tokens += public_tokens + content_tokens
        rows.append(
            {
                "skill": frontmatter.name,
                "frontmatter_tokens": public_tokens,
                "full_content_tokens": content_tokens,
                "full_injection_tokens": public_tokens + content_tokens,
            }
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "skill_token_benchmark.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "skill",
                "frontmatter_tokens",
                "full_content_tokens",
                "full_injection_tokens",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    saved = full_tokens - frontmatter_tokens
    saved_rate = saved / full_tokens if full_tokens else 0.0
    summary_path = RESULTS_DIR / "skill_token_benchmark_summary.md"
    summary_path.write_text(
        f"""# Skill Token Benchmark Summary

## Setup

- Skills: {len(rows)}
- Baseline: inject each Skill frontmatter and full content
- Progressive disclosure: inject frontmatter only, download full Skill on demand

## Results

| Metric | Tokens |
| --- | ---: |
| Full injection | {full_tokens} |
| Frontmatter only | {frontmatter_tokens} |
| Saved tokens | {saved} |

Saved token rate: {saved_rate:.1%}
""",
        encoding="utf-8",
    )

    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(f"Saved tokens: {saved}/{full_tokens} ({saved_rate:.1%})")


if __name__ == "__main__":
    main()

