"""Small shared helpers: logging, text normalisation, JSON snapshots."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

_SCRIPT_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.S | re.I)
_TAG_RE = re.compile(r"<[^>]+>")

_STOP = {"the", "a", "an", "of", "in", "on", "to", "for", "and", "is", "are",
         "at", "by", "with", "as", "from", "after", "over", "new", "says"}


# Windows hands a bare `python` a cp1252 stdout, which raises
# UnicodeEncodeError on anything this pipeline routinely handles: an emoji in
# a feed title, an Arabic or CJK headline, a box-drawing character in a
# banner. run.py forced UTF-8 on its own streams, but that only protected the
# one entry point — pytest, a direct `from pipeline import ...`, or any
# ad-hoc script bypassed it and died on a headline it merely tried to print.
# Doing it here covers every caller, and errors="replace" means logging can
# never be the thing that kills a run.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):        # already wrapped, or not a tty
        pass


# --- malware-distribution lures --------------------------------------------
# A GitHub search for new security/tooling repos returns, alongside real
# projects, throwaway accounts pushing "game cheats" and "Windows optimisers"
# whose README reads, verbatim:
#
#   "All-in-one loader ... download, extract, and run. Download the tool from
#    the button above. Extract the archive (password: respocket). Run as
#    Administrator"
#
# A password-protected archive exists to get past antivirus scanning, and
# "run as administrator" is the payload's request for privilege. These passed
# the star floors (10-80) on every search, so stars are being inflated and are
# no defence. Between 26 Oct and 2 Nov up to five of them reached the published
# page tagged as tools, with the archive password printed on the card —
# surfaced by GitGuardian flagging "respocket" as a leaked credential.
#
# TWO independent signals are required, because each alone has a legitimate
# twin. Researchers share samples as "password: infected" (Objective-See's own
# analyses on this very page say "malware collection (password: infect3d)
# ...please don't infect yourself!"), and admin tooling legitimately says
# "run as administrator". Only the lure says BOTH: here is an archive and its
# password, now run it elevated.
_LURE_ARCHIVE_PW = re.compile(
    r"\b(?:extract|unzip|unpack|unrar|archive|zip|rar|7z)\b[^.\n]{0,40}?"
    r"\(?\s*(?:password|pass|pwd)\s*[:=]\s*([^\s).,;]+)",
    re.I)
# The research convention: a sample shared deliberately, labelled as malware.
_RESEARCH_PASSWORDS = {"infected", "infect3d", "malware", "virus", "infect"}
_LURE_RUN = re.compile(
    r"\brun\s+(?:it\s+)?as\s+(?:an?\s+)?admin(?:istrator)?\b"
    r"|\ball[- ]in[- ]one\s+loader\b"
    r"|\bdownload\b[^.\n]{0,60}\b(?:extract|unzip)\b[^.\n]{0,60}\brun\b"
    r"|\bdownload\s+the\s+(?:tool|setup|installer|archive)\s+from\s+the\s+button\b",
    re.I)


def is_malware_lure(text: str) -> bool:
    """True when the text both hands over an archive password AND tells the
    reader to run what is inside it. See the block comment above."""
    if not text:
        return False
    pw = [m.group(1).lower().strip("'\"") for m in _LURE_ARCHIVE_PW.finditer(text)]
    if not [p for p in pw if p not in _RESEARCH_PASSWORDS]:
        return False
    return bool(_LURE_RUN.search(text))


def log(stage: str, msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{ts}] {stage:<12} {msg}", file=sys.stderr)


def banner(text: str) -> None:
    print(f"\n{'=' * 68}\n  {text}\n{'=' * 68}", file=sys.stderr)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def tokens(title: str) -> set[str]:
    """Content words from a headline, for similarity comparison."""
    words = re.findall(r"[a-z0-9']+", title.lower())
    return {w for w in words if w not in _STOP and len(w) > 2}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def overlap(a: set[str], b: set[str]) -> float:
    """Overlap coefficient: shared tokens over the SHORTER headline.

    Preferred over Jaccard for cross-outlet matching. Two outlets covering
    one story routinely write headlines of very different length, and
    Jaccard penalises that asymmetry hard enough to miss real matches
    (measured on live data: true pairs scored 0.33-0.60 on Jaccard but
    0.55-0.86 here).
    """
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def strip_html(text: str) -> str:
    """Feed blurbs routinely carry markup. It must never reach a summary."""
    import html as _html

    # Script/style bodies must go with their tags, not survive as text.
    text = _SCRIPT_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    return " ".join(_html.unescape(text).split())


def domain_of(url: str) -> str:
    from urllib.parse import urlsplit
    return urlsplit(url).netloc.lower().removeprefix("www.")


def _default(o: Any) -> Any:
    if isinstance(o, datetime):
        return o.isoformat()
    if is_dataclass(o):
        return asdict(o)
    if hasattr(o, "value"):        # Enum
        return o.value
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"not serialisable: {type(o)}")


def snapshot(obj: Any, path: Path, label: str = "") -> None:
    """Dump any stage's output to disk so a run can be replayed or diffed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=_default)
    if label:
        log("snapshot", f"{label} -> {path.name}")


def truncate(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "\u2026"


def headline(text: str, n: int) -> str:
    """Cut prose down to something that reads as a headline.

    A plain truncate is right for a body preview and wrong for a title.
    Social posts are the whole reason: a Telegram or Bluesky post is prose,
    so chopping it at 160 characters produced titles that stopped mid-word -
    "After a literal year of development, the WASM check patch has been
    merged into Anubis. This will be" - which is not a headline, it is a
    sentence someone interrupted.

    Prefer the first COMPLETE sentence. Otherwise break on the last whole
    word, so the worst case is still a clean cut rather than a severed one.
    """
    text = " ".join(text.split())
    if len(text) <= n:
        return text

    # min(40, n): interpolating n straight in built ".{40,30}" for any limit
    # under 40, which is not a valid repeat and raised re.error instead of
    # truncating. truncate() next door accepts any n, and so should this.
    m = re.match(r"(.{%d,%d}?[.!?])(?:\s|$)" % (min(40, n), n), text)
    if m:
        return m.group(1).strip()

    cut = text[:n].rstrip()
    space = cut.rfind(" ")
    if space > n * 0.6:                  # do not strand a tiny fragment
        cut = cut[:space].rstrip()
    return cut.rstrip(",;:-") + "\u2026"
