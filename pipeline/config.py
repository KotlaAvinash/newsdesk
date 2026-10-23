"""Config loading. One place that knows where files live."""

from __future__ import annotations

import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
SITE_DIR = ROOT / "site"

for _d in (DATA_DIR, RAW_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=None)
def _load(name: str) -> dict[str, Any]:
    with open(CONFIG_DIR / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def settings() -> dict[str, Any]:
    return _load("settings.yaml")


def feeds() -> dict[str, list[dict[str, Any]]]:
    return _load("feeds.yaml")


def interests() -> dict[str, Any]:
    return _load("interests.yaml")


ENV_FILE = ROOT / ".env"

# Groups whose read access to a secrets file is a problem. Anchored on a
# backslash or line start so "Users" does not also match "Authenticated Users"
# twice, and so any machine/domain prefix is accepted.
#   * boundary is line-start, a backslash, OR whitespace, because icacls puts
#     the first entry on the same line as the filename
#   * longest alternative first, or "Authenticated Users:" matches the bare
#     "Users" branch and gets mislabelled
_ACL_RISKY = re.compile(
    r"(?:^|\\|\s)(Authenticated Users|Everyone|Users):", re.MULTILINE)


def _warn_if_world_readable(path: Path) -> None:
    """Say something if the secrets file is readable by other local accounts.

    Found on this machine: .env inherited read access for the local Users
    group and Authenticated Users from its parent folder, meaning any account
    on the box could read the API key. Gitignoring a file protects it from
    the remote, not from the local filesystem.

    This only warns — silently changing permissions on a user's files would be
    worse than telling them.
    """
    try:
        if os.name == "nt":
            import subprocess
            out = subprocess.run(["icacls", str(path)], capture_output=True,
                                 text=True, timeout=10).stdout
            # Match the account NAME only. icacls prefixes vary by machine
            # and locale — BUILTIN\Users, MACHINE\Users, or a bare Users —
            # so anchoring on any one prefix silently never matches.
            risky = sorted({
                m.group(1) for m in _ACL_RISKY.finditer(out)
            })
            if risky:
                print(f"WARNING: {path.name} is readable by "
                      f"{', '.join(risky)} — other accounts on this machine "
                      f"can read your API keys.", file=sys.stderr)
                print(f'  Fix:  icacls "{path}" /inheritance:r '
                      f'/grant:r "%USERNAME%:(R,W)"', file=sys.stderr)
        else:
            mode = path.stat().st_mode
            if mode & 0o077:
                print(f"WARNING: {path.name} is group/world readable "
                      f"({oct(mode & 0o777)}). Run: chmod 600 {path}",
                      file=sys.stderr)
    except Exception:
        pass          # a permissions check must never break the run


def _load_dotenv() -> dict[str, str]:
    """Read KEY=value pairs from .env, if it exists.

    Secrets live here rather than in config/*.yaml for one reason: .env is
    gitignored and the YAML files are committed. A key in settings.yaml would
    be pushed to GitHub the first time the workflow ran.

    The real environment always wins, so GitHub Actions injecting a secret
    overrides whatever a stray local .env happens to contain.
    """
    if not ENV_FILE.exists():
        return {}

    _warn_if_world_readable(ENV_FILE)

    out: dict[str, str] = {}
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        # Tolerate the shapes people actually paste: quotes, spaces, export.
        key = key.strip().removeprefix("export ").strip()
        out[key] = value.strip().strip('"').strip("'")
    return out


def clean_secret(value: str | None, name: str = "secret") -> str | None:
    """Strip whitespace/control characters from a credential before use.

    This is the single fix for three separate leak paths, all with the same
    root cause. A token containing a control character — a key pasted across
    a line wrap is the realistic case — makes http.client raise

        ValueError("Invalid header value %r" % value)

    where %r is the FULL header, i.e. `Bearer <token>` or the Gemini key.
    That exception is then caught and logged:

        s1b_github.py   caught by `except Exception` and written to stderr
        llm_gemini.py   escapes as an uncaught traceback
        s8_notify.py    Telegram puts the token in the URL path, so
                        InvalidURL embeds it and three handlers echo it

    Patching each handler would leave the next one to be written exposed.
    Sanitising at the single point of entry closes all of them, and a token
    with a stray newline was never going to authenticate anyway.
    """
    if value is None:
        return None
    cleaned = "".join(ch for ch in value.strip() if ord(ch) > 0x20)
    if cleaned != value.strip():
        print(f"WARNING: {name} contained whitespace or control characters; "
              f"they were stripped. Check the value in .env is on one line.",
              file=sys.stderr)
    return cleaned or None


# --dry-run promises "run everything, write nothing". It was only ever
# honoured by stage 7 and stage 8: the summariser still called the API and
# still wrote summaries.json, and Budget still wrote usage.json on every
# call. So reproducing a bug with --dry-run spent the day's free-tier quota
# and left the counters claiming it had been spent by the real run, which
# then started from a stale number. _tally already fixed its own instance of
# this; these are the rest. One flag, set once, consulted by every writer.
WRITES_ENABLED = True


def set_dry_run() -> None:
    """Switch off every cache write and every paid API call for this process."""
    global WRITES_ENABLED
    WRITES_ENABLED = False


def writes_enabled() -> bool:
    return WRITES_ENABLED


def api_key() -> str | None:
    """LLM key: real environment first, then .env. Never from committed config."""
    name = settings()["llm"]["api_key_env"]
    raw = os.environ.get(name) or _load_dotenv().get(name)
    return clean_secret(raw, name)
