"""Tests for the fetcher's retry/fallback budget in fetch.common.

Why this exists: government legal sites block plain User-Agents, ship broken
TLS chains, and drop connections — often all three on the same URL. The UA and
TLS fallbacks must NOT consume the retry budget. When they did, a host needing
both had one real attempt left, so a single flaky connection failed the URL
outright and silently dropped a source from the corpus. Nothing downstream
reports a missing document, which is what makes this failure mode expensive.

Run:  .venv/Scripts/python.exe tests/test_fetch_retry.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests  # noqa: E402

from fetch import common  # noqa: E402

SSL = requests.exceptions.SSLError("broken certificate chain")
CONN = requests.exceptions.ConnectionError("connection reset")
GOV = "https://elibrary.judiciary.gov.ph/thebookshelf/showdocs/28/20426"
OTHER = "https://example.com/doc.html"


class FakeResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code
        self.content = b"ok"
        self.text = "ok"
        self.encoding = "utf-8"

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(str(self.status_code))


def scripted(sequence, counter):
    """Replace requests.get with one that replays `sequence` call by call."""
    it = iter(sequence)

    def fake_get(url, headers=None, timeout=None, verify=None):
        counter["n"] += 1
        counter["verify"] = verify
        item = next(it)
        if isinstance(item, Exception):
            raise item
        return FakeResponse(item)

    return fake_get


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    return ok


def run() -> int:
    results = []
    real_get = requests.get
    try:
        # The regression: UA block, then broken TLS, then one flaky connection.
        # The old budget was exhausted by the two fallbacks and this failed.
        counter = {"n": 0}
        requests.get = scripted([403, SSL, CONN, 200], counter)
        common._get_with_retry(GOV, 5)
        results.append(check(
            "survives UA block + TLS fallback + one flaky retry", True,
            f"(attempts: {counter['n']})"))

        # Fallbacks must actually take effect, not just be skipped.
        results.append(check("TLS verification was relaxed for the gov host",
                             counter["verify"] is False))

        # The genuine retry budget is still bounded.
        counter = {"n": 0}
        requests.get = scripted([CONN, CONN, CONN], counter)
        try:
            common._get_with_retry(OTHER, 5)
            results.append(check("gives up after 3 genuine failures", False))
        except requests.exceptions.ConnectionError:
            results.append(check("gives up after 3 genuine failures",
                                 counter["n"] == 3, f"(attempts: {counter['n']})"))

        # A non-gov.ph host must never get the insecure-TLS fallback.
        counter = {"n": 0}
        requests.get = scripted([SSL, SSL, SSL], counter)
        try:
            common._get_with_retry(OTHER, 5)
            results.append(check("no TLS relaxation for non-gov.ph hosts", False))
        except requests.exceptions.SSLError:
            results.append(check("no TLS relaxation for non-gov.ph hosts",
                                 counter["verify"] is not False))

        # Each fallback fires at most once — a server that always 403s must
        # terminate rather than switch headers forever.
        counter = {"n": 0}
        requests.get = scripted([403, 403, 403, 403], counter)
        try:
            common._get_with_retry(OTHER, 5)
            results.append(check("repeated 403s terminate", False))
        except requests.exceptions.HTTPError:
            results.append(check("repeated 403s terminate rather than loop",
                                 True, f"(attempts: {counter['n']})"))
    finally:
        requests.get = real_get

    failed = results.count(False)
    print(f"\n{len(results) - failed}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run())
