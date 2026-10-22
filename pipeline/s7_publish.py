"""Stage 7 — publish.

Writes the single artefact the website reads. Rejected items are written
to a separate audit file, never to the site, so a bad call can always be
traced after the fact.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .config import CACHE_DIR, DATA_DIR, SITE_DIR, settings
from .models import Item, Priority, Verdict, canonical_url
from .s2_clean import save_seen
from .utils import domain_of, log, now_utc, snapshot

NEWS_JSON = DATA_DIR / "news.json"
REJECTS_JSON = DATA_DIR / "rejected.json"
# A running tally across every run there has ever been. news.json only ever
# describes TODAY, so without this the page can say "40 stories" for months
# and never answer "how much has this thing actually found for me".
TOTALS_JSON = CACHE_DIR / "totals.json"

_ORDER = {Priority.CRITICAL: 0, Priority.IMPORTANT: 1, Priority.MINOR: 2}


def _resolve_links(items: list[Item]) -> None:
    """Turn Google News redirects into real publisher URLs, in place.

    Runs here rather than at harvest because it costs two HTTP requests per
    link, and the results are cached forever.

    BOUNDED, because "only what is being published" stopped meaning 40 items.
    When the publish cap went from 40 to everything in the window this became
    ~1,150 items' worth of links, and a rebuild sat for forty minutes at 1.2
    seconds of CPU — blocked on network, one redirect at a time, on course to
    blow the workflow's 75-minute budget.

    Items arrive ranked, so the cap spends the requests on the stories nearest
    the top of the page. An unresolved Google News link is not broken: it
    still redirects correctly in a browser, it just shows news.google.com as
    the source until a later run resolves it from cache.
    """
    from . import gnews

    limit = int((settings().get("publish") or {}).get("max_link_resolutions", 150))
    targets, seen = [], set()
    for item in items:                       # already ranked best-first
        for s in item.sources:
            u = s["url"]
            if gnews.is_gnews(u) and u not in seen:
                seen.add(u)
                targets.append(u)
        if len(targets) >= limit:
            break
    if not targets:
        return
    log("publish", f"resolving {len(targets)} Google News links (cap {limit})")

    resolved = gnews.resolve_many(targets)
    fixed = 0
    for item in items:
        for src in item.sources:
            real = resolved.get(src["url"])
            if real:
                src["url"] = real
                # Keep domain in step with url. It may still hold the
                # synthetic "<publisher>.publisher" placeholder derived from
                # the Google News title, which would then disagree with the
                # link shown on the same card.
                src["domain"] = domain_of(real)
                fixed += 1
    if fixed:
        log("publish", f"{fixed} reference links resolved to publishers")


def _next_run(cron: str, delay_hours: float = 0.0) -> str | None:
    """When the page will next actually refresh, as UTC ISO8601.

    NOT the cron time. GitHub queues scheduled workflows behind paid
    capacity: measured across 12 runs of this repo the delay from cron to
    finished run had a median of 5.5h, a minimum of 1.1h and a maximum of
    18.8h. Printing the cron time told the reader the page was overdue every
    single morning for five hours while it was doing exactly what it always
    does. `delay_hours` shifts the estimate to the observed reality.

    Only the daily form is handled, because that is the only form the
    workflow uses. Anything else returns None rather than guessing — a wrong
    "next update" time is worse than none, since the whole point is telling
    the reader whether the page is stale or simply between runs.
    """
    from datetime import timedelta

    parts = cron.split()
    if len(parts) != 5 or parts[2:] != ["*", "*", "*"]:
        return None
    try:
        minute, hour = int(parts[0]), int(parts[1])
    except ValueError:
        return None

    now = now_utc()
    nxt = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    nxt += timedelta(hours=delay_hours)
    if nxt <= now:
        nxt += timedelta(days=1)
    return nxt.isoformat()


def _apply_quotas(ranked: list, wcfg: dict) -> list:
    """Cap each category, then refill spare slots by global rank.

    A straight top-N cut let one high-volume wire take 39 of 40 slots on a
    live run. Categories are capped first so every section has something,
    then whatever is left over is filled by the best remaining items.
    """
    total = wcfg["max_items_published"]
    cap = wcfg["per_category_max"]
    floor = wcfg.get("per_category_min", 0)

    by_cat: dict[str, list] = {}
    for item in ranked:
        by_cat.setdefault(item.category, []).append(item)

    chosen, counts = [], {}
    picked: set[int] = set()

    # Reserve the floor first. A cap alone does not stop one section crowding
    # out another: when the topic scans flooded cyber and AI, markets was
    # squeezed to a single item even though its cap was 12.
    for cat, group in by_cat.items():
        for item in group[:floor]:
            if len(chosen) >= total:
                break
            chosen.append(item)
            picked.add(id(item))
            counts[cat] = counts.get(cat, 0) + 1

    for item in ranked:                      # already ranked priority-then-blend
        if id(item) in picked:
            continue
        n = counts.get(item.category, 0)
        if n < cap and len(chosen) < total:
            chosen.append(item)
            picked.add(id(item))
            counts[item.category] = n + 1

    if len(chosen) < total:                  # spare capacity: best of the rest
        for item in ranked:
            if len(chosen) >= total:
                break
            if id(item) not in picked:
                chosen.append(item)

    chosen.sort(key=lambda i: (_ORDER[i.priority], -i.blend))
    return chosen


def _tally(published: int, held: int, generated_at: str,
           persist: bool = True) -> dict | None:
    """Add this run to the lifetime totals and return them, or None.

    Keyed by run date so a re-run on the same day corrects that day's figure
    instead of double-counting it — reruns happen (a failed push, a manual
    trigger), and a total that inflates every time you retry is worse than
    no total at all.

    `persist=False` computes without touching the file, which is what a dry
    run needs.

    NOTHING in here may raise. This is a counter: it is not load-bearing, it
    is written before news.json, and an exception would abort stage 7 and
    lose the whole run to a corrupt file that nothing actually depends on.
    Every failure returns None and the caller simply omits the totals.
    """
    import json

    try:
        state = {"runs": {}, "first_run": generated_at}
        if TOTALS_JSON.exists():
            try:
                loaded = json.loads(TOTALS_JSON.read_text(encoding="utf-8"))
                # json.loads happily returns a list, a string or None for
                # input that is valid JSON but not an object — a truncated
                # write, a hand edit, a half-committed file from an
                # interrupted run. setdefault() on any of those raises
                # AttributeError, which is not a JSONDecodeError.
                if isinstance(loaded, dict):
                    state = loaded
                else:
                    log("publish", "! totals.json is not an object, "
                                   "starting a new tally")
            except (json.JSONDecodeError, OSError, ValueError):
                log("publish", "! totals.json unreadable, starting a new tally")

        runs = state.setdefault("runs", {})
        if not isinstance(runs, dict):
            runs = state["runs"] = {}
        first = state.setdefault("first_run", generated_at)
        if not isinstance(first, str) or len(first) < 10:
            first = state["first_run"] = generated_at

        runs[generated_at[:10]] = {"published": published, "held": held}
        # .get on both, not just one: an entry missing "published" is exactly
        # as likely as one missing "held", and indexing would raise.
        rows = [r for r in runs.values() if isinstance(r, dict)]
        totals = {
            "runs": len(rows),
            "since": first[:10],
            "published": sum(int(r.get("published", 0) or 0) for r in rows),
            "held": sum(int(r.get("held", 0) or 0) for r in rows),
        }
        if persist:
            TOTALS_JSON.write_text(json.dumps(state, indent=1), encoding="utf-8")
        return totals
    except Exception as exc:                      # never take down a run
        log("publish", f"! tally failed ({exc}); publishing without totals")
        return None


def _held_back(dead: list[Item]) -> list[dict]:
    """Rejected stories, carried onto the page but marked and defanged.

    These used to vanish into rejected.json. Most of them are perfectly
    real - 16 of 24 on a typical run are simply older than their section's
    lookback window, which says nothing at all about whether the reporting
    is sound - and every one of them names its source, so the reader can
    judge for themselves.

    The ones rejected for an UNSUPPORTED CLAIM are different, and cannot be
    shipped as they stand. What failed there is our own summary: the model
    wrote a sentence the verifier could not trace back to the article. The
    story may be fine; the prose about it is not evidence. So the generated
    text is stripped from those and only the headline, the source and the
    reason survive. Publishing an unverifiable summary under a "held back"
    label would still be publishing it.
    """
    out: list[dict] = []
    for item in dead:
        d = item.to_json()
        reason = (item.reject_reason or "").lower()
        stale = "lookback window" in reason
        d["held_back"] = True
        d["verdict"] = Verdict.UNVERIFIED.value
        d["priority"] = Priority.MINOR.value
        if not stale:
            # Model-written prose that failed its own check. Drop it rather
            # than caveat it - a label does not make a claim traceable.
            for field in ("summary", "one_liner", "bottom_line"):
                d[field] = ""
            d["key_facts"] = []
            d["claims"] = []
        out.append(d)
    return out


def run(items: list[Item], dry_run: bool = False) -> dict:
    cfg = settings()

    live = [i for i in items if i.verdict is not Verdict.REJECTED]
    dead = [i for i in items if i.verdict is Verdict.REJECTED]

    # Within a priority band, rank by the blended score, not trust alone —
    # sorting by trust buried every interesting story under official notices.
    live.sort(key=lambda i: (
        _ORDER[i.priority],
        -i.blend,
        -(i.published.timestamp() if i.published else 0),
    ))
    live = _apply_quotas(live, cfg["window"])

    sched = cfg.get("schedule", {})
    # Record the uids of the links AS HARVESTED, before resolution rewrites
    # them. Stage 2 tests the raw harvested url against the seen store, so
    # storing the resolved publisher url instead would mean a Google News
    # item could never match and never be suppressed.
    seen_uids = {
        hashlib.sha1(canonical_url(s["url"]).encode()).hexdigest()[:16]
        for i in live for s in i.sources
    }

    _resolve_links(live)

    payload = {
        "generated_at": now_utc().isoformat(),
        "next_update": _next_run(sched.get("cron_utc", ""),
                                 float(sched.get("typical_delay_hours", 0))),
        "schedule": {
            "cron_utc": sched.get("cron_utc", ""),
            "display_tz": sched.get("display_tz", "UTC"),
            "display_name": sched.get("display_name", ""),
        },
        "window_hours": cfg["window"]["lookback_hours"],
        "per_category_hours": cfg["window"].get("per_category_hours", {}),
        "counts": {
            "published": len(live),
            "rejected": len(dead),
            "total_shown": len(live) + len(dead),
            "critical": sum(1 for i in live if i.priority is Priority.CRITICAL),
            "important": sum(1 for i in live if i.priority is Priority.IMPORTANT),
            "verified": sum(1 for i in live if i.verdict is Verdict.VERIFIED),
            "corroborated": sum(1 for i in live if i.verdict is Verdict.CORROBORATED),
            "established": sum(1 for i in live if i.verdict is Verdict.ESTABLISHED),
            "unverified": sum(1 for i in live if i.verdict is Verdict.UNVERIFIED),
            "disputed": sum(1 for i in live if i.verdict is Verdict.DISPUTED),
            "new": sum(1 for i in live if i.is_new),
        },
        "items": [i.to_json() for i in live] + _held_back(dead),
    }
    # persist=not dry_run: --dry-run is documented as "run everything, write
    # nothing", and this used to rewrite totals.json before the guard below
    # ever ran — so debugging the pipeline permanently inflated the lifetime
    # figure the page prints, and the workflow committed it.
    totals = _tally(len(live), len(dead), payload["generated_at"],
                    persist=not dry_run)
    if totals:
        payload["totals"] = totals

    if dry_run:
        log("publish", f"DRY RUN - would publish {len(live)}, reject {len(dead)}")
        return payload

    snapshot(payload, NEWS_JSON, "news")
    # The site is served as a plain static directory, so its data file lives
    # beside index.html rather than being fetched from ../data.
    snapshot(payload, SITE_DIR / "news.json", "site copy")

    # data/ keeps the FULL record, prose and all — it is the local audit trail
    # and is not served to anyone.
    snapshot({"generated_at": payload["generated_at"],
              "items": [i.to_json() for i in dead]}, REJECTS_JSON, "rejects")

    # site/ gets the SAME defanging news.json gets. _held_back strips the
    # model prose from claim-failure items precisely because "a label does not
    # make a claim traceable" — and then this file published it verbatim
    # anyway. Measured on the committed output: 16 of 26 held-back items had
    # summary == "" in news.json and a full 507-873 character summary, a
    # bottom_line and two claims in rejected.json. The policy was enforced in
    # one file and defeated in the other, and rejected.json is the one the
    # page's "not published" counter opens.
    snapshot({"generated_at": payload["generated_at"],
              "items": _held_back(dead)},
             SITE_DIR / "rejected.json", "site rejects")

    # Mark published sources as seen so tomorrow's run skips them.
    # Both forms: the harvested url so stage 2 can match it next run, and the
    # resolved url so a direct hit on the publisher is recognised too.
    seen_uids |= {
        hashlib.sha1(canonical_url(s["url"]).encode()).hexdigest()[:16]
        for i in live for s in i.sources
    }
    save_seen(seen_uids)

    log("publish", f"{len(live)} published, {len(dead)} rejected, "
                   f"{len(seen_uids)} urls marked seen")
    return payload
