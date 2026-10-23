"""Stage 1 — harvest.

Pull every feed in the registry, in parallel, and return raw Articles
inside the lookback window. Network failures never abort the run; a dead
feed is logged and skipped.
"""

from __future__ import annotations

import html
import ssl
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import feedparser

from . import gnews
from .config import feeds, settings
from .models import Article
from .utils import domain_of, log, strip_html

# Windows Python ships no usable CA store, so several perfectly valid feeds
# (CISA, Anthropic, TechCrunch) fail with CERTIFICATE_VERIFY_FAILED. certifi
# supplies a current bundle. Verification stays ON — this fixes the trust
# store, it does not disable the check.
try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()


def _parse_date(entry) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        st = entry.get(key)
        if st:
            return datetime(*st[:6], tzinfo=timezone.utc)
    return None


# Below this, a feed's <content> block is a teaser and the article still has
# to be fetched. Calibrated on the 1,050 cluster leads that came from a feed
# in the frozen harvest of 2026-10-22:
#
#   content >= 1000 chars   94 of 94 fetchable leads had a feed block at
#                           least 60% as long as the fetched article
#                           (median ratio 1.05, p10 0.98) — 0 teasers
#   content 400-999 chars   11 of 25 leads fetched a body over 2000 chars,
#                           i.e. the feed block WAS a teaser
#
# So 1000 is where "the feed shipped the article" stops being a guess.
FULL_TEXT_MIN_CHARS = 1000


def _feed_text(entry) -> tuple[str, str]:
    """(blurb for the card, full article text if the feed shipped it).

    Atom feeds carry the article in `content` and a teaser in `summary`, and
    only `summary` was being read — so stage 3 spent an HTTP request
    re-downloading text already sitting in the response. Measured across the
    120 feeds in this registry on 2026-10-22: 938 entries ship a `content`
    block of 1000+ characters, and for 31 feeds that block is the whole
    article while the summary is a stub —

        The Register Security   median content 3,338 vs summary    78
        Elastic Security Labs   median content 15,563 vs summary  182
        SpecterOps              median content 12,306 vs summary  468
        Securelist              median content 23,363 vs summary  203
        Recorded Future         median content 6,457 vs summary   167

    101 of 1,168 cluster leads had such a block, which is 101 HTTP requests
    that no longer need making — and 7 of them were Ars Technica, which
    returns 403/405 to every user-agent tried and had a 0/16 fetch rate, so
    the feed's own copy is the only copy obtainable at all.

    The blurb stays capped at 1200 characters. Stage 4 keyword-scans
    `summary_raw`, so letting a 20,000-character article in through that
    field would change scoring; the full text goes to `body`, which is what
    stage 3 fills and stages 5-6 read.
    """
    best = ""
    for block in entry.get("content") or []:
        value = strip_html(block.get("value") or "")
        if len(value) > len(best):
            best = value
    summary = strip_html(entry.get("summary") or "")
    # Some feeds carry only `content`; without the fallback those entries had
    # an EMPTY blurb, so a card whose fetch failed showed nothing at all.
    blurb = (summary or best)[:1200]
    return blurb, (best if len(best) >= FULL_TEXT_MIN_CHARS else "")


class _Redirect308(urllib.request.HTTPRedirectHandler):
    """Follow HTTP 308 Permanent Redirect.

    urllib handles 301/302/303/307 but not 308, so a feed that moved with a
    308 raised HTTPError and was recorded as dead. Two feeds hit this
    (Intigriti, ProjectDiscovery) and both are perfectly alive — the redirect
    just was not being followed.
    """

    def http_error_308(self, req, fp, code, msg, headers):
        return self.http_error_301(req, fp, 301, msg, headers)


_OPENER = urllib.request.build_opener(_Redirect308)


def _fetch_bytes(url: str, cfg: dict, user_agent: str | None = None) -> bytes:
    """Fetch one feed. `user_agent` overrides the global one for this feed.

    The global UA is a browser string because PIB and TechCrunch 403 a plain
    bot — but the same string is itself a 403 at some hosts. Huntress is the
    measured case, reproduced three times on 2026-10-23:

        chrome UA (the project default)  HTTP 403, every attempt
        plain reader UA                  HTTP 200, 742 entries,
                                         newest 14h old, 7 inside its window

    One global answer cannot be right for both, so the registry can say what
    a particular host wants. No cloaking involved in either direction: both
    strings say what they are, the volume is one request per feed per run.
    """
    req = urllib.request.Request(
        url,
        headers={"User-Agent": user_agent or cfg["harvest"]["user_agent"],
                 "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
                 "Cache-Control": "no-cache"},
    )
    # _OPENER, not urlopen: it carries the 308 handler above.
    import ssl as _ssl
    https = urllib.request.HTTPSHandler(context=_SSL_CTX)
    opener = urllib.request.build_opener(_Redirect308, https)
    with opener.open(req, timeout=cfg["harvest"]["timeout_seconds"]) as resp:
        return resp.read()


def window_for(category: str, cfg: dict, spec: dict | None = None) -> int:
    """Lookback for one feed.

    Three levels, most specific first: a per-feed `window_hours`, then the
    category window, then the global default.

    The per-feed level exists because publishing cadence is a property of the
    SOURCE, not its category. Deep research labs — watchTowr, Assetnote,
    Doyensec, Include Security, Sonar — publish once or twice a month, and
    every one of them returned zero items under a 168h category window even
    though the feeds were verified working. A month-old teardown of a
    pre-auth RCE has not stopped being worth reading.
    """
    if spec and spec.get("window_hours"):
        return int(spec["window_hours"])
    per = cfg["window"].get("per_category_hours", {})
    return per.get(category, cfg["window"]["lookback_hours"])


def _fetch_one(category: str, spec: dict, cutoff: datetime, cfg: dict) -> list[Article]:
    parsed = feedparser.parse(
        _fetch_bytes(spec["url"], cfg, spec.get("user_agent")))
    if getattr(parsed, "bozo", 0) and not parsed.entries:
        log("harvest", f"! {spec['name']}: {getattr(parsed, 'bozo_exception', 'no entries')}")
        return []

    out: list[Article] = []
    for entry in parsed.entries[: cfg["harvest"]["max_per_feed"]]:
        link = entry.get("link") or ""
        # Feeds escape their titles, and nothing downstream decoded them, so
        # cards rendered "Brazilian &amp; Global Financial Systems" and
        # "GoPro says it&#8217;s still committed". unescape twice: a few feeds
        # double-encode, which leaves "&amp;amp;" after one pass.
        title = html.unescape(html.unescape(
            (entry.get("title") or "").strip())).strip()
        if not link or not title:
            continue
        # Google News wraps every story in a redirect and names the real
        # publisher in the title. Attribute it here, for free, so clustering
        # and outlet-counting see the true source instead of "news.google.com".
        src_name, src_domain, aggregator = spec["name"], domain_of(link),             bool(spec.get("aggregator", False))
        if gnews.is_gnews(link):
            found = gnews.publisher_of(title)
            if found:
                src_name, src_domain = found
                title = gnews.strip_publisher(title)

        published = _parse_date(entry)
        # Undated entries are kept — stage 4 re-checks the date from the page.
        if published and published < cutoff:
            continue
        blurb, full = _feed_text(entry)
        out.append(
            Article(
                url=link,
                title=title,
                source=src_name,
                domain=src_domain,
                category=category,
                tier=spec.get("tier", "C"),
                published=published,
                summary_raw=blurb,
                # Normally stage 3's job, but when the feed already shipped
                # the article there is nothing left to fetch. Stage 3 skips
                # any article that arrives with a body. See _feed_text.
                body=full,
                is_primary=bool(spec.get("primary", False)),
                is_aggregator=aggregator,
                window_hours=int(spec.get("window_hours") or 0),
            )
        )
    log("harvest", f"  {spec['name']:<24} {len(out):>3} in window")
    return out


def run() -> list[Article]:
    cfg = settings()
    now = datetime.now(timezone.utc)
    registry = feeds()

    jobs = [(cat, spec) for cat, specs in registry.items() for spec in specs]
    windows = {cat: window_for(cat, cfg) for cat in registry}
    log("harvest", f"{len(jobs)} feeds, windows " +
        ", ".join(f"{c}={h}h" for c, h in windows.items()))

    articles: list[Article] = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {
            pool.submit(_fetch_one, c, s,
                        now - timedelta(hours=window_for(c, cfg, s)), cfg): s
            for c, s in jobs
        }
        for fut in as_completed(futures):
            try:
                articles.extend(fut.result())
            except Exception as exc:                      # a feed dying is normal
                log("harvest", f"! {futures[fut]['name']}: {exc}")

    log("harvest", f"total {len(articles)} articles")
    return articles
