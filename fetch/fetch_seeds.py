"""Download the statutes listed in seeds.txt into corpus/_fetched/.

Usage:
  python fetch/fetch_seeds.py                 # fetch all from seeds.txt
  python fetch/fetch_seeds.py --file my.txt   # use a different seed file
  python fetch/fetch_seeds.py --force         # re-download even if present

After it finishes, run:  python cli.py ingest
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fetch import common  # noqa: E402


def parse_seeds(path: Path) -> list[tuple[str, str]]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" not in line:
            print(f"  [skip] malformed line (no '|'): {line}")
            continue
        name, url = line.split("|", 1)
        entries.append((name.strip(), url.strip()))
    return entries


def main():
    ap = argparse.ArgumentParser()
    default_seeds = Path(__file__).resolve().parent / "seeds.txt"
    ap.add_argument("--file", default=str(default_seeds),
                    help="Seed file (name | url per line)")
    ap.add_argument("--subdir", default="lawphil",
                    help="Subfolder under corpus/_fetched to save into "
                         "(e.g. lawphil, prc, doh)")
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if already fetched")
    args = ap.parse_args()

    seeds = parse_seeds(Path(args.file))
    print(f"{len(seeds)} seed URLs to fetch into "
          f"{common.FETCHED_DIR / args.subdir}\n")

    ok, failed, skipped = 0, 0, 0
    for name, url in seeds:
        if not args.force and common.already_fetched(name, subdir=args.subdir):
            print(f"  [have] {name}")
            skipped += 1
            continue
        try:
            path = common.download(name, url, subdir=args.subdir)
            print(f"  [ok]   {name}  ->  {path.name}")
            ok += 1
            common.polite_sleep()
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL] {name}  ({e})")
            failed += 1

    print(f"\nDone. {ok} fetched, {skipped} already present, {failed} failed.")
    if ok:
        print("Next:  python cli.py ingest")


if __name__ == "__main__":
    main()
