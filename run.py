#!/usr/bin/env python
"""newsdesk — one batch run: harvest -> validate -> publish.

    python run.py                 full run, writes data/news.json
    python run.py --dry-run       run everything, write nothing
    python run.py --stop-after 2  stop after a given stage (1-7)
    python run.py --no-extract    skip full-text fetch (fast, blurbs only)
    python run.py --show          pretty-print the last published news.json
    python run.py --update-blocklist   refresh the Iffy low-credibility list
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

# The Windows console defaults to cp1252, which cannot encode the box-drawing
# and arrow characters used below. Force UTF-8 on both streams.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from pipeline import (s1_harvest, s1b_github, s1c_bluesky, s1d_telegram,
                      s1e_vulns,
                      s2_clean, s3_extract,
                      s4_corroborate, s5_summarize, s6_verify, s6b_judge,
                      s7_publish, s8_notify)
from pipeline.config import RAW_DIR
from pipeline.models import Priority, Verdict
from pipeline.s2_clean import BLOCKLIST_PATH
from pipeline.s7_publish import NEWS_JSON
from pipeline.utils import banner, log, snapshot, truncate

# Iffy Index of Unreliable Sources — domains MBFC rates low/very-low factual.
# Published as a public Google Sheet (CC-licensed); this is its CSV export.
IFFY_URL = ("https://docs.google.com/spreadsheets/d/"
            "1ck1_FZC-97uDLIlvRJDTrGqBk0FuDe9yHkluROgpGS8/"
            "gviz/tq?tqx=out:csv&sheet=Iffy-news")

BADGE = {
    Priority.CRITICAL:  "\033[91m[CRITICAL] \033[0m",
    Priority.IMPORTANT: "\033[93m[IMPORTANT]\033[0m",
    Priority.MINOR:     "\033[90m[minor]    \033[0m",
}
MARK = {
    Verdict.VERIFIED:      "\033[92mVERIFIED\033[0m",
    Verdict.CORROBORATED:  "\033[96mcorroborated\033[0m",
    Verdict.ESTABLISHED:   "established",
    Verdict.UNVERIFIED:    "\033[93munverified\033[0m",
    Verdict.DISPUTED:      "\033[95mDISPUTED\033[0m",
    Verdict.REJECTED:      "\033[91mREJECTED\033[0m",
}


def update_blocklist() -> None:
    """Fetch the Iffy.news index of low-credibility domains."""
    log("blocklist", f"fetching {IFFY_URL}")
    try:
        req = urllib.request.Request(IFFY_URL, headers={"User-Agent": "newsdesk/0.1"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8", "replace")
    except Exception as exc:
        log("blocklist", f"! failed: {exc}")
        log("blocklist", "download the CSV manually to data/cache/iffy_blocklist.txt")
        return

    import csv, io

    domains = set()
    reader = csv.DictReader(io.StringIO(text))
    # Column is "Domain" in the published sheet; fall back to first column.
    for row in reader:
        raw = (row.get("Domain") or next(iter(row.values()), "") or "").strip().lower()
        raw = raw.removeprefix("http://").removeprefix("https://").removeprefix("www.")
        raw = raw.split("/")[0]
        if "." in raw and " " not in raw:
            domains.add(raw)

    BLOCKLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    BLOCKLIST_PATH.write_text(
        "# Iffy.news index — domains rated low / very-low factual by MBFC\n"
        + "\n".join(sorted(domains)),
        encoding="utf-8",
    )
    log("blocklist", f"cached {len(domains)} domains")


def test_llm() -> int:
    """Prove the key and model work before spending a whole run on them."""
    from pipeline.config import api_key, settings
    from pipeline import llm_gemini

    cfg = settings()["llm"]
    key = api_key()
    if not key:
        env = cfg["api_key_env"]
        print(f"{env} is not set in this shell.", file=sys.stderr)
        print(f'  setx {env} "your-key"   (then open a NEW terminal)',
              file=sys.stderr)
        return 1

    model = cfg["tiers"][0]["model"]
    print(f"key found ({len(key)} chars), calling {model}...", file=sys.stderr)
    try:
        one_liner = llm_gemini.smoke_test(model, key)
    except llm_gemini.GeminiError as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 1
    print("\nOK - the model replied. Its one-liner for the test article:",
          file=sys.stderr)
    print(f"  {one_liner}", file=sys.stderr)
    return 0


def show_last() -> None:
    if not NEWS_JSON.exists():
        print("no news.json yet — run: python run.py")
        return
    data = json.loads(NEWS_JSON.read_text(encoding="utf-8"))
    render(data)


def render(payload: dict) -> None:
    c = payload["counts"]
    banner(f"NEWSDESK  ·  {payload['generated_at'][:16].replace('T', ' ')} UTC  ·  "
           f"{payload['window_hours']}h window")
    print(f"  {c['published']} published   {c['critical']} critical   "
          f"{c['verified']} verified   {c['corroborated']} corroborated   "
          f"{c['unverified']} unverified   "
          f"{c['rejected']} rejected\n")

    # Items arrive ranked by priority, so grouping on a change of category
    # would print the same heading over and over. Bucket first, then render.
    groups: dict[str, list[dict]] = {}
    for item in payload["items"]:
        groups.setdefault(item["category"], []).append(item)

    for category, entries in groups.items():
        label = category.replace("_", " ").upper()
        print(f"\n\033[1m── {label} {'─' * max(2, 48 - len(label))}\033[0m")
        for item in entries:
            _render_card(item)
    print()


def _render_card(item: dict) -> None:
    pr = Priority(item["priority"])
    vd = Verdict(item["verdict"])
    print(f"\n{BADGE[pr]} {MARK[vd]}  ·  {item['corroboration']} outlet(s)"
          f"  ·  trust {item['trust_score']}")
    print(f"  \033[1m{truncate(item['title'], 100)}\033[0m")
    if item["one_liner"]:
        print(f"  \033[36m→ {truncate(item['one_liner'], 130)}\033[0m")
    if item["summary"]:
        print(f"    {truncate(item['summary'], 300)}")
    srcs = " · ".join(s["name"] for s in item["sources"][:5])
    print(f"    \033[90msources: {srcs}\033[0m")
    print(f"    \033[90m{item['sources'][0]['url']}\033[0m")
    if item["primary_links"]:
        print(f"    \033[92mprimary: {item['primary_links'][0]}\033[0m")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="write nothing to disk")
    ap.add_argument("--stop-after", type=int, default=8, metavar="N",
                    help="stop after stage N (6=verify, 7=judge, 8=publish)")
    ap.add_argument("--no-extract", action="store_true", help="skip full-text fetch")
    ap.add_argument("--no-social", action="store_true",
                    help="skip the GitHub and Bluesky collectors (RSS only)")
    ap.add_argument("--show", action="store_true", help="print last published news.json")
    ap.add_argument("--update-blocklist", action="store_true")
    ap.add_argument("--test-llm", action="store_true",
                    help="one tiny Gemini call to prove the key works")
    ap.add_argument("--test-telegram", action="store_true",
                    help="verify the Telegram bot token and channel access")
    ap.add_argument("--quiet", action="store_true", help="no console render")
    args = ap.parse_args()

    if args.dry_run:
        # Before anything imports a Budget or loads a cache.
        from pipeline.config import set_dry_run
        set_dry_run()
        log("run", "DRY RUN - no disk writes, no API calls")

    if args.update_blocklist:
        update_blocklist()
        return 0
    if args.test_llm:
        return test_llm()
    if args.test_telegram:
        return s8_notify.selftest()
    if args.show:
        show_last()
        return 0

    banner("STAGE 1  harvest")
    articles = s1_harvest.run()
    # Non-RSS collectors. Each is independent and self-disabling via config,
    # so a GitHub rate limit or a Bluesky outage costs those items and nothing
    # else — the RSS harvest has already succeeded by this point.
    if not args.no_social:
        articles += s1b_github.run()
        articles += s1c_bluesky.run()
        articles += s1d_telegram.run()
        articles += s1e_vulns.run()
        log("harvest", f"total with collectors: {len(articles)}")
    if not articles:
        print("no articles harvested — check network / feeds.yaml", file=sys.stderr)
        return 1
    snapshot(articles, RAW_DIR / "s1_articles.json", "articles")
    if args.stop_after == 1:
        return 0

    banner("STAGE 2  clean + cluster")
    clusters = s2_clean.run(articles)
    snapshot(clusters, RAW_DIR / "s2_clusters.json", "clusters")
    if args.stop_after == 2:
        return 0

    banner("STAGE 3  extract")
    if args.no_extract:
        log("extract", "skipped (--no-extract)")
        for cl in clusters:
            for a in cl.articles:
                a.body = a.summary_raw
    else:
        clusters = s3_extract.run(clusters)
    bodies = {cl.key: cl.lead.body for cl in clusters}
    if args.stop_after == 3:
        return 0

    banner("STAGE 4  corroborate + score")
    items = s4_corroborate.run(clusters)
    snapshot(items, RAW_DIR / "s4_items.json", "scored items")
    if args.stop_after == 4:
        return 0

    banner("STAGE 5  summarize")
    items = s5_summarize.run(items, bodies)
    # Stage 5 replaces the keyword interest score with the model's importance
    # judgement, so priority has to be recomputed or that judgement is ignored.
    items = s4_corroborate.rescore(items)
    # The model's importance score reorders the ranking, so items it promoted
    # into the publish set may never have been summarised. Fill those gaps.
    items = s5_summarize.top_up(items, bodies)
    items = s4_corroborate.rescore(items)
    snapshot(items, RAW_DIR / "s5_items.json", "summarised items")
    if args.stop_after == 5:
        return 0

    banner("STAGE 6  verify (the gate)")
    items = s6_verify.run(items, bodies)
    if args.stop_after == 6:
        return 0

    banner("STAGE 6b  independent judge")
    items = s6b_judge.run(items, bodies)
    if args.stop_after <= 7:   # 6b is the judge; 7 stops before publish
        return 0

    banner("STAGE 7  publish")
    payload = s7_publish.run(items, dry_run=args.dry_run)

    banner("STAGE 8  notify")
    s8_notify.run(items, payload["generated_at"], dry_run=args.dry_run)

    if not args.quiet:
        render(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
