"""Stage 1b — new and fast-rising tools from GitHub.

RSS tells you what journalists wrote about; it does not tell you what got
built. This uses the GitHub search API, which needs no authentication for
this volume, to surface repositories that appeared or gained traction inside
the window.

Two different questions are asked, because they surface different things:

  * `created:>DATE`  — brand new projects (the "new tool" case)
  * `pushed:>DATE` with a star floor — established projects shipping again

A star floor matters more here than anywhere else in the pipeline: GitHub
search returns thousands of new repos a week, almost all of them abandoned
scaffolding, coursework, or AI-generated filler.
"""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

from .config import settings
from .models import Article
from .utils import headline, log, strip_html

try:
    import certifi
    _SSL = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL = ssl.create_default_context()

API = "https://api.github.com/search/repositories"


def _search(query: str, cfg: dict, per_page: int, token: str | None) -> list[dict]:
    url = f"{API}?{urllib.parse.urlencode({'q': query, 'sort': 'stars', 'order': 'desc', 'per_page': per_page})}"
    headers = {
        "User-Agent": cfg["harvest"]["user_agent"],
        "Accept": "application/vnd.github+json",
    }
    # Optional: raises the search rate limit from 10/min to 30/min.
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=cfg["harvest"]["timeout_seconds"],
                                    context=_SSL) as resp:
            return json.load(resp).get("items", [])
    except urllib.error.HTTPError as exc:
        # 403 is the unauthenticated rate limit (10 searches/min), not an auth
        # failure. 422 means the query itself was rejected — worth showing in
        # full, because it is a config bug rather than a transient one.
        detail = ""
        if exc.code == 422:
            detail = f" — {exc.read().decode('utf-8', 'replace')[:120]}"
        log("github", f"! HTTP {exc.code} for '{query[:52]}'{detail}")
        return []
    except Exception as exc:
        log("github", f"! {exc} — skipped")
        return []


def _to_article(repo: dict, category: str, kind: str) -> Article:
    stars = repo.get("stargazers_count", 0)
    desc = strip_html(repo.get("description") or "").strip()
    topics = ", ".join(repo.get("topics", [])[:8])

    # The title carries the star count because that is the single most useful
    # signal when scanning a list of unfamiliar repository names.
    # headline(), not desc[:110]: a raw slice cut descriptions mid-word on the
    # page ("...monitoring, and infrastruct"). headline() ends on a sentence
    # or a whole word.
    title = f"{repo['full_name']} — {headline(desc, 110)}" if desc else repo["full_name"]

    body = (f"{desc}\n\nGitHub repository {repo['full_name']}, "
            f"{stars:,} stars, language {repo.get('language') or 'unspecified'}. "
            f"{'Topics: ' + topics + '. ' if topics else ''}"
            f"{'Newly created.' if kind == 'new' else 'Recently updated.'}")

    return Article(
        url=repo["html_url"],
        title=title,
        source=f"GitHub ({kind})",
        domain="github.com",
        category=category,
        tier="C",                 # a repo is a primary artifact but not vetted
        published=_parse(repo.get("created_at") if kind == "new"
                         else repo.get("pushed_at")),
        summary_raw=body,
        is_primary=False,
    )


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def run() -> list[Article]:
    cfg = settings()
    gh = cfg.get("github")
    if not gh or not gh.get("enabled", False):
        return []

    since = (datetime.now(timezone.utc)
             - timedelta(hours=gh["lookback_hours"])).strftime("%Y-%m-%d")

    from .config import clean_secret
    token = clean_secret(os.environ.get(gh.get("token_env", "")),
                         gh.get("token_env", "GITHUB_TOKEN"))
    if token:
        log("github", "using GITHUB_TOKEN (30 searches/min)")

    out: list[Article] = []
    seen: set[str] = set()
    pace = 0 if token else gh.get("pace_seconds", 7)

    for n, spec in enumerate(gh["searches"]):
        kind = spec.get("kind", "new")
        floor = spec.get("min_stars", 0)
        date_field = "created" if kind == "new" else "pushed"
        # ONE topic per search: GitHub rejects a query made only of OR'd
        # qualifiers, so `topic:a OR topic:b` is a 422 every time.
        query = f"topic:{spec['topic']} {date_field}:>{since} stars:>={floor}"

        if n and pace:
            time.sleep(pace)

        for repo in _search(query, cfg, gh["per_search"], token):
            if repo["html_url"] in seen:
                continue
            seen.add(repo["html_url"])
            out.append(_to_article(repo, spec["category"], kind))

    log("github", f"{len(out)} repositories from {len(gh['searches'])} searches")
    return out
