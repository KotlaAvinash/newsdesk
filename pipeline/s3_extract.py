"""Stage 3 — full-text extraction.

Optional. Without it the pipeline still runs on feed blurbs, but summary
quality and claim verification both degrade, so it is on by default when
trafilatura is installed.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

from . import netguard
from .config import settings
from .models import Cluster
from .utils import log

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


def run(clusters: list[Cluster], lead_only: bool = True) -> list[Cluster]:
    if not AVAILABLE:
        log("extract", "trafilatura not installed - using feed blurbs only")
        for cl in clusters:
            for art in cl.articles:
                art.body = art.summary_raw
        return clusters

    cfg = settings()
    timeout = cfg["harvest"]["timeout_seconds"]

    # Only the lead article of each cluster needs full text; the rest are
    # corroboration links, not summary inputs.
    targets = [cl.lead for cl in clusters] if lead_only else \
              [a for cl in clusters for a in cl.articles]

    # One HTTP fetch each, so this - not the number of published cards - is
    # the run's clock. Now that the page publishes everything in the window
    # instead of a top 40, fetching every lead would mean ~1,070 requests for
    # the sake of the ~150 a model will actually read. Rank cheaply and fetch
    # only that many; the rest keep the feed's own blurb, which is all a
    # headline-level card needs.
    limit = (cfg.get("extract") or {}).get("max_items")
    skipped = []
    if limit and len(targets) > limit:
        owner = {id(cl.lead): cl for cl in clusters}

        def worth(art):
            # No model scores exist yet here, so use what is already known:
            # whether it is a primary artifact, how many outlets carried it,
            # how good the source is, and how fresh it is.
            cl = owner.get(id(art))
            return (art.is_primary,
                    len(cl.articles) if cl else 1,
                    {"A": 3, "B": 2, "C": 1}.get(art.tier, 0),
                    art.published.timestamp() if art.published else 0)

        targets.sort(key=worth, reverse=True)
        targets, skipped = targets[:limit], targets[limit:]
        for art in skipped:
            art.body = art.summary_raw

    log("extract", f"fetching {len(targets)} article bodies"
                   + (f", {len(skipped)} kept as blurbs" if skipped else ""))
    ok = 0
    # Download in parallel, parse in series. See _parse for why.
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_download, a.url, timeout): a for a in targets}
        for fut in as_completed(futures):
            art = futures[fut]
            try:
                html = fut.result()
            except Exception as exc:
                log("extract", f"! {art.domain}: {exc}")
                html = ""
            try:
                body = _parse(html)
            except Exception as exc:                 # a bad page is not fatal
                log("extract", f"! parse {art.domain}: {str(exc)[:70]}")
                body = ""
            art.body = body or art.summary_raw
            ok += bool(body)

    log("extract", f"{ok}/{len(targets)} full bodies, rest fell back to blurbs")
    return clusters
