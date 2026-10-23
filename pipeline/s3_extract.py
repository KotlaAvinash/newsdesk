"""Stage 3 — full-text extraction.

Optional. Without it the pipeline still runs on feed blurbs, but summary
quality and claim verification both degrade, so it is on by default when
trafilatura is installed.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

from . import gnews, netguard
from .config import settings
from .models import Cluster
from .utils import domain_of, log

try:
    import trafilatura
    from trafilatura.settings import use_config
    AVAILABLE = True
except ImportError:                     # pipeline must survive without it
    AVAILABLE = False


@lru_cache(maxsize=1)
def _traf_config(timeout: int):
    """trafilatura's own settings, with OUR timeout and SSRF guard pinned.

    `harvest.timeout_seconds` was threaded all the way down to _download and
    then dropped on the floor — fetch_url was called with no config at all,
    so every request used trafilatura's DEFAULT_CONFIG instead. Measured
    against 2.3.1: DOWNLOAD_TIMEOUT is 30, double the 15 this project
    configures. Proven by timing a connect to a non-routable address —
    3.0s with the config, 21.0s without.

    That gap is the difference between a slow day and a lost one. At
    extract.max_items 1400 over 8 workers, 175 serial fetches per worker at
    30s each is ~87 minutes of timeouts alone, past the workflow's 75-minute
    cap — the job is killed and the whole run discarded.

    SSRF_PROTECTION is pinned on rather than merely left at its default.
    trafilatura 2.3.1 installs a pool whose connections vet the peer address
    AFTER connecting, on every hop, so a feed link that redirects to the
    cloud metadata endpoint is refused even though netguard only ever saw
    the first URL — and post-connect vetting also closes the DNS-rebinding
    window netguard's pre-connect resolve leaves open. Verified both ways
    against a private address that was actually listening: on -> blocked,
    off -> 78,345 bytes. It is load-bearing, so it is written down.
    """
    cfg = use_config()
    cfg.set("DEFAULT", "DOWNLOAD_TIMEOUT", str(timeout))
    cfg.set("DEFAULT", "SSRF_PROTECTION", "on")
    return cfg


def _download(url: str, timeout: int) -> str:
    """Fetch the raw HTML. Safe to run concurrently — this is network I/O."""
    # Article links come from feeds, so they are untrusted input. trafilatura
    # happens to reject file:// today, but relying on a third-party library's
    # incidental behaviour for a security property is not a control.
    reason = netguard.check(url)
    if reason:
        log("extract", f"! refusing {url[:52]}: {reason}")
        return ""
    return trafilatura.fetch_url(url, config=_traf_config(timeout)) or ""


def _parse(html: str) -> str:
    """Pull the article text out. MUST NOT run concurrently.

    trafilatura parses through lxml, which is libxml2 — a C extension. Running
    it across eight threads aborted the process outright:

        fetching 700 article bodies, 428 kept as blurbs
        double free or corruption (out)
        2351 Aborted (core dumped) python run.py --quiet

    That is a glibc heap abort, not a Python exception, so nothing upstream
    could catch it and the whole run died at stage 3. It survived for weeks at
    a 220-item cap and started failing the first night the cap was 700, which
    is what a concurrency race looks like: more work per run, same bug, and
    suddenly it is reached.

    The pool exists for network latency, not for CPU, so the fetch stays
    parallel and the parse comes back to the main thread. Costs a few seconds
    of wall clock and removes the race completely.
    """
    if not html:
        return ""
    return trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    ) or ""


def _worth(art, owner: dict) -> tuple:
    """Cheap ranking, used both for the fetch cap and the resolution cap.

    No model scores exist yet at this stage, so use what is already known:
    whether it is a primary artifact, how many outlets carried it, how good
    the source is, and how fresh it is.
    """
    cl = owner.get(id(art))
    return (art.is_primary,
            len(cl.articles) if cl else 1,
            {"A": 3, "B": 2, "C": 1}.get(art.tier, 0),
            art.published.timestamp() if art.published else 0)


def _spread(pairs: list[tuple]) -> list[tuple]:
    """Round-robin the fetch order across hosts, keeping each host's rank.

    Ranking the targets (so the caps spend themselves on the best stories)
    has a side effect: articles from one source share a tier, a publish time
    and usually a primary flag, so sorting lines them up CONSECUTIVELY and
    eight workers then hit one host with eight simultaneous requests and
    keep going. Measured, on the same frozen harvest, ranked but not spread:

        ndtv.com              20/20 bodies -> 0/20
        exploit-db.com          7/7 bodies -> 0/7
        technologyreview.com    5/5 bodies -> 0/5
        wiz.io                  6/6 bodies -> 1/6

    45 articles that extracted fine in cluster order stopped extracting —
    not one of them a failure of the extractor, all of them a host refusing
    a burst. Taking the best remaining item from each host in turn keeps the
    ranking intact and means consecutive requests go to different hosts.
    """
    buckets: dict[str, list] = {}
    for item in pairs:
        buckets.setdefault(domain_of(item[1]), []).append(item)
    out: list[tuple] = []
    while buckets:
        for host in list(buckets):
            out.append(buckets[host].pop(0))
            if not buckets[host]:
                del buckets[host]
    return out


def _resolve_aggregator_links(targets: list, limit: int) -> dict[str, str]:
    """Google News redirect token -> real publisher URL, for the best links.

    THIS IS THE WHOLE EXTRACTION PROBLEM. Measured on a frozen harvest of
    2026-10-22, 1,168 cluster leads:

        gnews links      315 targets    0 bodies      0.0%
        everything else  853 targets  810 bodies     95.0%

    Every single Google News target failed, and they were 315 of the 358
    failures (88%). The link is a redirect stub — the fetched page is a few
    hundred bytes of Google JavaScript with no article in it — so the request
    was guaranteed waste before it was sent.

    Resolving the token first turns that around. On 200 of those same
    targets, resolved through the existing pipeline.gnews resolver:

        resolution succeeded   238/238 attempted  (100%)
        extraction after it    184/200            (92.0%)
        median body            4,834 chars, against an 88-char feed blurb

    Costs two requests per link, which is why it is capped and why the
    results go in the permanent gnews cache that stage 7 already reads: a
    link resolved here is free when stage 7 rewrites the card's reference
    link later in the same run, so the two stages share one resolution
    rather than paying twice.

    Anything left unresolved is NOT fetched. A 0-for-315 measurement is not
    a reason to retry it harder.
    """
    todo = [a.url for a in targets if gnews.is_gnews(a.url)]
    if not todo or limit <= 0:
        return {}
    # targets arrive ranked, so a cap spends the requests on the best stories
    todo = list(dict.fromkeys(todo))[:limit]
    try:
        return gnews.resolve_many(todo)
    except Exception as exc:                   # never worth losing a run over
        log("extract", f"! link resolution failed: {str(exc)[:80]}")
        return {}


def run(clusters: list[Cluster], lead_only: bool = True) -> list[Cluster]:
    if not AVAILABLE:
        log("extract", "trafilatura not installed - using feed blurbs only")
        for cl in clusters:
            for art in cl.articles:
                art.body = art.summary_raw
        return clusters

    cfg = settings()
    timeout = cfg["harvest"]["timeout_seconds"]
    ex = cfg.get("extract") or {}

    # Only the lead article of each cluster needs full text; the rest are
    # corroboration links, not summary inputs.
    targets = [cl.lead for cl in clusters] if lead_only else \
              [a for cl in clusters for a in cl.articles]
    owner: dict[int, Cluster] = {}
    for cl in clusters:
        for art in cl.articles:
            owner.setdefault(id(art), cl)

    # Articles that already carry their text: the feed shipped the whole
    # article in <content> (s1_harvest._feed_text) or the record IS the
    # artifact (s1e_vulns). 107 of 1,169 leads on the measured run — 107
    # HTTP requests not made, and 7 of them Ars Technica, which answers
    # 403/405 to every user-agent tried and has never yielded a body.
    from_feed = [a for a in targets if a.body]
    targets = [a for a in targets if not a.body]

    # One HTTP fetch each, so this - not the number of published cards - is
    # the run's clock. Now that the page publishes everything in the window
    # instead of a top 40, fetching every lead would mean ~1,070 requests for
    # the sake of the ~150 a model will actually read. Rank cheaply and fetch
    # only that many; the rest keep the feed's own blurb, which is all a
    # headline-level card needs.
    limit = ex.get("max_items")
    targets.sort(key=lambda a: _worth(a, owner), reverse=True)
    skipped: list = []
    if limit and len(targets) > limit:
        targets, skipped = targets[:limit], targets[limit:]
        for art in skipped:
            art.body = art.summary_raw

    # Domains that answer, but never with an article. Each needs a measured
    # 0-for-N before it goes in; see the comment in settings.yaml.
    #
    # Checked BEFORE resolution as well as after, because a Google News
    # entry already knows its publisher — the title names it and
    # gnews.publisher_of maps it to a domain — so a wall we have measured
    # can be honoured without spending the two resolution requests first.
    # Worth 108 requests a run on Reuters alone.
    dead = {d.lower() for d in (ex.get("skip_domains") or [])}
    blocked = 0
    live = []
    for art in targets:
        if art.domain.lower() in dead:
            blocked += 1
            art.body = art.summary_raw
        else:
            live.append(art)

    resolved = _resolve_aggregator_links(
        live, int(ex.get("max_link_resolutions") or 0))

    fetch: list[tuple[object, str]] = []
    unresolved = 0
    for art in live:
        url = resolved.get(art.url, art.url)
        if gnews.is_gnews(url):          # token we could not resolve
            unresolved += 1
            art.body = art.summary_raw
            continue
        if domain_of(url) in dead:       # only knowable after resolution
            blocked += 1
            art.body = art.summary_raw
            continue
        fetch.append((art, url))

    fetch = _spread(fetch)
    log("extract", f"fetching {len(fetch)} article bodies; "
                   f"{len(from_feed)} already full-text from the feed, "
                   f"{unresolved} unresolved aggregator links and "
                   f"{blocked} never-yield domains not fetched"
                   + (f", {len(skipped)} over the cap" if skipped else ""))

    # A wall-clock floor under the stage. The per-URL ceiling is the timeout
    # plus trafilatura's own retry backoff, so a day where many hosts hang is
    # unbounded in aggregate — and this job runs inside a 75-minute workflow
    # that kills the whole run, publishing nothing, if stage 3 overruns.
    # Degrading to blurbs is strictly better than losing the day. Generous on
    # purpose: a healthy run of 1,168 fetches measured 74 seconds.
    budget = float(ex.get("max_seconds") or 900)
    ok = 0
    started = time.monotonic()
    # Download in parallel, parse in series. See _parse for why.
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_download, url, timeout): art
                   for art, url in fetch}
        try:
            for fut in as_completed(futures, timeout=budget):
                art = futures[fut]
                try:
                    html = fut.result()
                except Exception as exc:
                    log("extract", f"! {art.domain}: {exc}")
                    html = ""
                try:
                    body = _parse(html)
                except Exception as exc:             # a bad page is not fatal
                    log("extract", f"! parse {art.domain}: {str(exc)[:70]}")
                    body = ""
                art.body = body or art.summary_raw
                ok += bool(body)
        except TimeoutError:
            late = [a for f, a in futures.items() if not f.done()]
            for fut in futures:
                fut.cancel()
            for art in late:
                art.body = art.body or art.summary_raw
            log("extract", f"! stopped at the {budget:.0f}s budget with "
                           f"{len(late)} fetches outstanding; they keep blurbs")

    total = len(fetch) + len(from_feed)
    full = ok + len(from_feed)
    log("extract", f"{full}/{total} full bodies in "
                   f"{time.monotonic() - started:.0f}s "
                   f"({ok} fetched, {len(from_feed)} from feeds), "
                   f"rest fell back to blurbs")
    return clusters
