"""Split the fetched Rules of Court (Rules 72-109) into one file per RULE.

Why this exists: LawPhil serves Part II as a single 148k-character page, and
indexed whole it is close to useless. Measured before the split:

  * 199 chunks from that one file, sharing only 25 distinct section labels —
    "Section 1." appeared 28 times, "Section 2." 27 times, because every rule
    restarts its own numbering.
  * The chunk carrying the answer to "what is the deadline for a claim against
    the estate" (Rule 86 Sec. 2, the six-to-twelve month statute of non-claims)
    was labelled just "Section 2." with law=None — ambiguous 27 ways and
    UNCITABLE.
  * 199 near-identically-labelled chunks compete with each other, so the right
    one loses to its own siblings. Retrieval surfaced the file but the wrong
    section of it, and the answer denied that any deadline existed.

One file per rule gives each a real identity ("RULE 86 - Claims Against
Estate"), which ragmed.chunking turns into a citable `law`, and drops the
per-file chunk count to something where a section label means one thing.

Idempotent: re-running rewrites the per-rule files. The combined source file is
removed at the end so it is not indexed twice; re-fetching the seed restores it,
so run this again after any re-fetch.

Usage:
  python fetch/split_rules_of_court.py
  python cli.py ingest
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fetch import common  # noqa: E402

SOURCE = (common.FETCHED_DIR / "lawphil" /
          "rules-of-court-part-ii-special-proceedings-rules-72-to-109-"
          "claims-against-estate.txt")
OUT_DIR = common.FETCHED_DIR / "lawphil" / "rules-of-court"

# "RULE 86" alone on its line; the human-readable title is the next non-empty
# line ("Claims Against Estate"). A few rules carry no title.
RULE_RE = re.compile(r"^[ \t]*RULE[ \t]+(\d+)[ \t]*$", re.MULTILINE)


def _title_after(text: str, pos: int) -> str:
    for line in text[pos:pos + 400].splitlines():
        line = line.strip()
        if line:
            return line if len(line) <= 120 else ""
    return ""


def split(source: Path = SOURCE, out_dir: Path = OUT_DIR) -> list[Path]:
    if not source.exists():
        raise SystemExit(
            f"Not found: {source}\nFetch it first:\n  python fetch/fetch_seeds.py "
            f"--file fetch/seeds_billing.txt --subdir lawphil")

    text = source.read_text(encoding="utf-8")
    matches = list(RULE_RE.finditer(text))
    if not matches:
        raise SystemExit("No 'RULE <n>' headings found — has the page layout "
                         "changed? Refusing to write anything.")

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for i, m in enumerate(matches):
        number = m.group(1)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.start():end].strip()
        title = _title_after(text, m.end())

        # The header is what makes the file citable: ragmed.chunking reads the
        # document's identity out of its first lines, and every chunk of this
        # file then carries "Rule 86" instead of a bare "Section 2."
        label = f"RULE {number}" + (f" - {title}" if title else "")
        header = (f"Rules of Court of the Philippines\n"
                  f"PART II - SPECIAL PROCEEDINGS\n"
                  f"{label}\n\n")
        name = f"rules-of-court-rule-{number}" + (
            f"-{common.slugify(title, maxlen=60)}" if title else "")
        path = out_dir / f"{name}.txt"
        path.write_text(header + body, encoding="utf-8")
        written.append(path)

    return written


def main() -> None:
    written = split()
    print(f"Wrote {len(written)} per-rule files to {OUT_DIR}")
    for p in written[:3]:
        print(f"  {p.name}")
    print("  ...")
    for p in written[-2:]:
        print(f"  {p.name}")

    # Remove the combined file so the same text is not indexed twice: duplicate
    # chunks split the score between two copies and crowd the top-k.
    SOURCE.unlink()
    print(f"\nRemoved the combined source file ({SOURCE.name}).")
    print("Next:  python cli.py ingest")


if __name__ == "__main__":
    main()
