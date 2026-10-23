"""Stage 5 — summarise, write the one-liner, score importance.

Provider is chosen in settings.yaml:

  none    deterministic extractive fallback; the whole pipeline stays
          testable with no API key at all
  gemini  Gemini Developer API, free tier

Whatever the provider, the contract is the same: summary, one_liner,
importance 1-5, and claims each carrying the verbatim span that supports
them. Claims exist so stage 6 can verify them — a provider that cannot
produce them cannot be verified, and its output must stay untrusted.

Summaries are cached by cluster identity. The dashboard shows a rolling
48h window, so the same article legitimately appears in several runs; it
should be paid for once.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .config import CACHE_DIR, api_key, interests, settings
from . import tagging
from .budget import Budget
from .models import Claim, Item, Verdict
from .utils import log, truncate

SUMMARY_CACHE = CACHE_DIR / "summaries.json"
# Entries, not bytes. At ~1.5 KB a summary this is roughly 7.5 MB.
CACHE_MAX = 5000


# ---------------------------------------------------------------- cache ----

def _load_cache() -> dict[str, dict]:
    if SUMMARY_CACHE.exists():
        try:
            loaded = json.loads(SUMMARY_CACHE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, ValueError):
            log("summarize", "! summary cache corrupt, starting fresh")
            return {}
        # json.loads returns a list, a string or None just as happily for
        # input that is valid JSON but not an object — a truncated write, a
        # hand edit, a half-committed file from an interrupted run. The very
        # next thing done with this is `cache[ckey] = ...`, which raises
        # TypeError on a list and kills stage 5 after stages 1-4 have already
        # spent the whole network budget. _tally learned this the hard way;
        # the same guard belongs here.
        if not isinstance(loaded, dict):
            log("summarize", "! summary cache is not an object, starting fresh")
            return {}
        return loaded
    return {}


def _save_cache(cache: dict[str, dict]) -> None:
    """Bound the cache, evicting what has NOT been used rather than what was
    written longest ago.

    The trim was `list(cache.items())[-5000:]`, described as "keep the most
    recent entries". Dict order is INSERTION order, and re-assigning an
    existing key does not move it, so a summary reused every single day kept
    its original position and was evicted as soon as 5,000 newer entries
    arrived — while 5,000 write-once entries survived. Exactly backwards, and
    every eviction of a hot entry is a summary re-bought tomorrow.

    Measured on the committed file: 5,000 entries, at the cap, 7.8 MB, with
    only 4,907 distinct payloads — 93 slots spent on duplicates left behind
    when a cluster re-keyed. A probe re-filing one hot entry behind 6,000
    cold ones confirmed it did not survive the trim.

    `_touch` now moves an entry to the end on every hit, so the tail really
    is the least recently used.
    """
    if len(cache) > CACHE_MAX:
        for key in list(cache)[:len(cache) - CACHE_MAX]:
            del cache[key]
    SUMMARY_CACHE.write_text(json.dumps(cache, ensure_ascii=False),
                             encoding="utf-8")


def _touch(cache: dict[str, dict], old_key: str, new_key: str) -> None:
    """Re-file a cache hit under the current key, at the END of the order.

    Also drops the old key when it differs, instead of leaving a duplicate
    behind — that is where the 93 duplicate payloads came from, and each one
    accelerates the eviction above.
    """
    payload = cache.pop(old_key)
    cache[new_key] = payload


def _apply(item: Item, payload: dict, source: str, body: str = "") -> None:
    item.summary = payload.get("summary", "")
    item.one_liner = payload.get("one_liner", "")
    item.bottom_line = payload.get("bottom_line", "")
    # Cap at five: past that the fact list competes with the summary instead
    # of supporting it.
    item.key_facts = [f.strip() for f in payload.get("key_facts", [])
                      if isinstance(f, str) and f.strip()][:5]
    tagging.apply(item, body, payload.get("tags"))
    # A real summary replaces an extractive one, so drop the marker or the
    # top-up pass would keep re-selecting the same items forever.
    item.trace = [t for t in item.trace
                  if "extractive" not in t and "heuristic" not in t]
    item.claims = [
        Claim(text=c.get("text", ""), support_span=c.get("support_span", ""))
        for c in payload.get("claims", [])
    ]
    imp = payload.get("importance")
    if isinstance(imp, int) and 1 <= imp <= 5:
        # The model's judgement replaces the keyword heuristic outright. The
        # heuristic cannot tell a central-bank penalty from a scheduled
        # treasury auction; this is the whole reason the LLM stage exists.
        item.interest_score = int((imp - 1) / 4 * 100)
        item.note(f"importance {imp}/5 from {source}")

    # LAST, and only if the response actually carried prose. Setting this on
    # the first line marked an item "model" even when the payload parsed but
    # was empty — and top_up() selects gaps by `summary_source != "model"`
    # while the site suppresses the "unsummarised" chip for the same value.
    # So a blank card became invisible to both the pass meant to rescue it
    # and the label meant to disclose it, while implying it had been
    # summarised and checked.
    if item.summary.strip() or item.bottom_line.strip():
        item.summary_source = "model"


# ----------------------------------------------------------- fallback ----

def _vuln_facts(item: Item, body: str) -> list[str]:
    """Structured facts for a vulnerability, pulled out with no model call.

    A CVE card carries its own facts in plain text — the identifier, the
    severity, the affected range, the fixed version — and they sit in a
    predictable shape because they come from NVD, CISA KEV or a CVE feed.
    Before this, an unsummarised CVE shipped as a bare headline while the
    numbers a reader actually needs were sitting in the body unparsed.

    Deterministic, so it costs nothing and cannot hallucinate: every value
    returned is copied out of the source text, never inferred.
    """
    text = f"{item.title} {body}"

    # EVERYTHING here is gated on a real CVE identifier. Without this the
    # "Affected" and "Patched in" patterns match ordinary prose: a markets
    # story reading "revenue prior to 2019 was unaudited ... the firm updated
    # in 2023" produced the bullets "Affected: prior to 2019" and "Patched
    # in: 2023", which is meaningless and looks like a security advisory.
    cve = re.search(r"CVE-\d{4}-\d{4,7}", text)
    if not cve:
        return []

    facts: list[str] = []

    # The score must sit beside an explicit CVSS marker, or be joined to its
    # label by whitespace alone as the NVD collector formats it
    # ("CVE-1234-5678 (CRITICAL 9.8)"). Accepting any number within twelve
    # characters of a severity word turned "HIGH: 10 products affected" into
    # "CVSS 10, HIGH" and "Rated LOW; 10 vendors" into "CVSS 10, LOW" — the
    # latter self-contradictory, since CVSS 10 is CRITICAL by definition.
    sev = re.search(
        r"CVSS[^0-9]{0,25}(10(?:\.0)?|[0-9]\.[0-9])"
        r"(?:[^A-Za-z0-9]{0,4}\(?(CRITICAL|HIGH|MEDIUM|LOW))?",
        text, re.I)
    if not sev:
        sev = re.search(r"(CRITICAL|HIGH|MEDIUM|LOW)\s+(10(?:\.0)?|[0-9]\.[0-9])",
                        text, re.I)

    line = cve.group(0)
    if sev:
        groups = [g for g in sev.groups() if g]
        score = next((g for g in groups if any(c.isdigit() for c in g)), "")
        label = next((g for g in groups if g.isalpha()), "")
        if score and label:
            line += f": CVSS {score}, {label.upper()}"
        elif score:
            line += f": CVSS {score}"
    facts.append(f"Identifier: {line}")

    aff = re.search(r"((?:versions?\s+)?(?:prior to|before|earlier than|up to"
                    r"(?: and including)?)\s+v?[0-9][0-9A-Za-z.\-]*)", text, re.I)
    if aff:
        facts.append(f"Affected: {aff.group(1).strip().rstrip(chr(46))}")

    fix = re.search(r"(?:fixed|patched|resolved|remediated|updated)\s+in\s+"
                    r"(?:version\s+)?(v?[0-9][0-9A-Za-z.\-]*)", text, re.I)
    if fix:
        facts.append(f"Patched in: {fix.group(1).rstrip(chr(46))}")

    if re.search(r"known ransomware campaign use:\s*known", text, re.I):
        facts.append("Ransomware: used in known ransomware campaigns")
    due = re.search(r"Federal remediation due:\s*(\d{4}-\d{2}-\d{2})", text, re.I)
    if due:
        facts.append(f"CISA remediation due: {due.group(1)}")
    if re.search(r"actively exploited|exploitation in the wild|"
                 r"KEV catalogue|KEV catalog", text, re.I):
        facts.append("Status: exploitation reported in the wild")

    return facts[:4]


_BOILERPLATE = re.compile(
    r"^\s*(?:Posted by .{0,60}?on \w{3} \d{1,2}"          # seclists / oss-sec
    r"|=+|-{4,}|_{4,}"                                     # ASCII rules
    r"|(?:Read|View) (?:more|the full article)\b"
    r"|Click here\b|Subscribe\b|Share this\b"
    r"|The post .{0,80}? appeared first on\b"              # WordPress footer
    r"|This (?:article|post) (?:originally )?appeared\b)",
    re.I)

# Feeds that publish a teaser and cut it off mid-thought, then append their
# own advert. Medium is the worst: the blurb ends "... Continue reading on
# Medium »", which is not part of the article at all.
_TEASER_TAIL = re.compile(
    r"\s*(?:\.{3}|…)?\s*Continue reading on .{0,40}?(?:»|>>|$)"
    r"|\s*Continue reading\.{0,3}\s*$"
    r"|\s*\[\.{3}\]\s*$", re.I)

# Scripts this page does not render summaries in. The site is written in
# English and a reader cannot use a summary they cannot read; the headline
# and the source link still get published so nothing is hidden.
_NON_LATIN = re.compile(
    r"[Ѐ-ӿ"      # Cyrillic
    r"֐-׿"       # Hebrew
    r"؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿"   # Arabic
    r"ऀ-ॿ"       # Devanagari
    r"฀-๿"       # Thai
    r"぀-ヿ"       # Kana
    r"一-鿿"       # CJK
    r"가-힯]")     # Hangul


def _mostly_non_english(text: str) -> bool:
    """True when the text is not usable by an English-reading audience.

    A ratio, not a single character: a legitimate English story routinely
    names a Chinese company or quotes a Russian handle, and dropping those
    would lose real reporting. Only when a THIRD of the letters are another
    script is the summary genuinely unreadable here.
    """
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 20:
        return False
    return sum(bool(_NON_LATIN.match(c)) for c in letters) / len(letters) > 0.33


def _clean_sentence(s: str) -> str:
    """Strip the furniture a feed wraps around its actual text."""
    s = re.sub(r"https?://\S+", "", s)          # bare URLs read as noise
    s = _TEASER_TAIL.sub("", s)
    # Exploit-DB and packetstorm prefix every entry with a metadata header
    # written as shell comments — "# Exploit Title: ... # Date: ... # Exploit
    # Author: ... # CVE: ..." — which has no sentence terminator, so the
    # splitter kept it as one long "sentence" and it became the whole card.
    # The prose starts after "# Description:".
    m = re.search(r"#\s*Description\s*:\s*", s, re.I)
    if m:
        s = s[m.end():]
    s = re.sub(r"#\s*(?:Exploit Title|Date|Exploit Author|Author|Vendor"
               r"(?: Homepage)?|Software Link|Version|CVE|Tested on|Category|"
               r"Platform|Type)\s*:[^#]*", "", s, flags=re.I)
    s = re.sub(r"\s*#+\s*", " ", s)
    return " ".join(s.split()).strip(" -–—|·")


def _heuristic(item: Item, body: str) -> None:
    """No-LLM fallback. Honest about what it is: extractive, not abstractive."""
    # Mailing lists and WordPress feeds bury the story under furniture. Taking
    # the first three sentences verbatim produced card bodies like "Posted by
    # Alan Coopersmith on Sep 04 https://github.com/... The GHSA advisories
    # listed above can be found on https://..." — a byline, two URLs and a
    # cross-reference, which is not a summary of anything.
    raw = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body)]
    sentences = []
    for s in raw:
        if _BOILERPLATE.match(s):
            continue
        s = _clean_sentence(s)
        # After stripping URLs a "sentence" can collapse to nothing useful.
        if len(s) <= 40 or not re.search(r"[a-z]{3}", s):
            continue
        if _mostly_non_english(s):
            continue
        sentences.append(s)
    # Was: "(no article text available - headline only)". Invisible at a
    # cap of 40 published cards; at ~1,150 it is roughly 166 cards a day
    # whose entire body is an apology string, which reads as the page
    # being broken. The card already renders cleanly with no summary and
    # the "unsummarised" chip already says what it is, so say nothing.
    # Five sentences at 700, not three at 480. Extractive bodies ran to a
    # median of 141 characters against 438 for model-written ones, which
    # is the difference between a summary and a caption. Now that most
    # items have real article text behind them there is something to take.
    item.summary = truncate(" ".join(sentences[:5]), 700) if sentences else ""

    # Was: "Cyber Attacks: single report, unconfirmed." on 909 of 1,146 cards
    # — 79% of the page carrying a line that restated the card's own section
    # heading and the verdict chip already sitting above it, in the slot a
    # reader looks to for what the story MEANS. Saying nothing there is more
    # informative than saying something empty.
    item.one_liner = ""

    # These are the article's own sentences quoted back, so "entailed" is
    # true by construction and means nothing. Keeping them under the heading
    # "Claims checked against the source" would have read as verification on
    # most of the page once it started publishing everything in the window.
    # No check happened here, so no claims are reported.
    item.claims = []
    item.bottom_line = ""      # only the model can judge what the point is
    # A vulnerability carries its own facts in the text. Lift them out rather
    # than shipping a bare headline while the CVE id, the severity and the
    # fixed version sit unread in the body.
    item.key_facts = _vuln_facts(item, body)
    tagging.apply(item, body)  # keyword tags still work with no model
    item.summary_source = "extract"
    item.note("summary: the article's opening sentences, not model-written "
              "and not fact-checked")


def _fallback_all(items: list[Item], bodies: dict[str, str], why: str) -> list[Item]:
    log("summarize", why)
    for item in items:
        _heuristic(item, bodies.get(item.cluster_key, ""))
    return items


def _shortlist(items: list[Item], limit: int, per_category: int,
               multiplier: float = 2.0) -> tuple[list[Item], list[Item]]:
    """Split into the items worth an API call and the rest.

    Category-aware, and it has to be. Stage 7 fills a per-category quota, so
    a purely global top-N shortlist starves the quieter sections: on a live
    run that left 24 of 40 published cards with no model summary at all,
    because they were only published to fill their category's quota.

    So each category gets its own allocation first — generously, since the
    model's importance score can reorder within a category — and any slots
    left over are filled by global rank.

    Rejected items are never sent: paying to summarise something the gate has
    already thrown out is pure waste.
    """
    live = [i for i in items if i.verdict is not Verdict.REJECTED]
    dead = [i for i in items if i.verdict is Verdict.REJECTED]

    by_cat: dict[str, list[Item]] = {}
    for item in sorted(live, key=lambda i: -i.blend):
        by_cat.setdefault(item.category, []).append(item)

    allocation = max(1, int(per_category * multiplier))
    chosen: list[Item] = []
    picked: set[int] = set()
    # ROUND-ROBIN, not category-by-category. Filling one category's whole
    # allocation before starting the next only works while every allocation
    # fits inside the limit; the moment it does not, the categories that
    # happen to come first eat the entire budget and the rest get nothing.
    # With five categories, a 150 budget and a 60 allocation that is exactly
    # what happened - three categories took all 150 and markets and
    # india_world shipped with no model summaries at all.
    for rank in range(allocation):
        if len(chosen) >= limit:
            break
        for group in by_cat.values():
            if len(chosen) >= limit:
                break
            if rank < len(group):
                chosen.append(group[rank])
                picked.add(id(group[rank]))

    for item in sorted(live, key=lambda i: -i.blend):
        if len(chosen) >= limit:
            break
        if id(item) not in picked:
            chosen.append(item)
            picked.add(id(item))

    rest = [i for i in live if id(i) not in picked]
    return chosen, rest + dead



BULK_DOMAINS = {"bsky.app", "github.com"}


def _is_bulk(item) -> bool:
    """Should this item be summarised by a cheap, high-quota model?

    Two cases, both the user's point: material that does not update daily
    (social posts, repository listings) and material ranked too low to be
    read closely. Neither justifies spending one of twenty daily flagship
    requests, but both are still worth collecting and summarising.
    """
    from .models import Priority

    if item is None:
        return False
    if any(s.get("domain") in BULK_DOMAINS for s in item.sources):
        return True
    return item.priority is Priority.MINOR


def _generate(batch, budget, key, profile, tag_list, llm_gemini,
              n: int, total: int, prefer: str = "premium"):
    """Try each model with quota until one serves this batch.

    A model can fail for reasons that say nothing about the batch — a 503
    overload, a 429 quota wall. Falling straight back to an extractive
    summary in that case wastes an entire ladder of working models, which is
    exactly what happened on a live run when gemini-3.7-flash returned 503
    while seven other models sat idle.

    Returns (results, model_name), or (None, "") if every tier is spent.
    """
    # Models that have already failed on THIS batch. A bench is 45s but a 503
    # takes up to 93s to surface, so a benched model was eligible again the
    # moment this loop came round and the same tier was picked over and over
    # until its strikes ran out — all three flagship attempts a run landed on
    # batch 1. The ladder's premise is that a different model is likelier to
    # answer, so each model gets at most one go per batch.
    tried: set[str] = set()
    while True:
        # The batch size goes in so a tier that SLICES the batch is only
        # chosen when it has enough calls for the whole thing. Without it a
        # gemma tier with one call of allowance left still took a 20-item
        # batch and made all seven, overdrawing per_run by six.
        tier = budget.next_model(prefer, len(batch), skip=tried)
        if tier is None:
            # "No tier available" and "no tier left today" are not the same
            # thing. A bench is 45 seconds; abandoning the batch is for the
            # rest of the run. Every batch after the first all-benched moment
            # used to fall to an extractive summary in the same instant —
            # measured 12 of 20 batches served on 2026-10-22 with eight dying
            # inside one 45s window. Wait for the first model to come back.
            if budget.wait_out_cooldown():
                # Having deliberately waited out the shortest bench, walking
                # the ladder again from the top is the whole point, so the
                # per-batch skip list starts over. Termination still holds:
                # strikes per tier and the total wait are both bounded.
                tried.clear()
                continue
            return None, ""

        # A tier with a tighter token ceiling takes the batch in slices.
        chunk = tier.batch_size or len(batch)
        slices = [batch[i:i + chunk] for i in range(0, len(batch), chunk)]

        try:
            results = []
            for part in slices:
                budget.wait(tier)
                payload = [{k: b[k] for k in ("id", "title", "source", "body")}
                           for b in part]
                got = llm_gemini.call(tier.model, key, profile, payload,
                                      tags=tag_list)
                # Record AFTER success. Debiting first meant a 429 or 503 —
                # which returns nothing — still spent the local allowance,
                # so the counter drifted pessimistic against the provider.
                budget.record(tier)
                results += got
            return results, tier.model

        except llm_gemini.RateLimited as exc:
            # Quota refusals last until the quota resets, so remember them.
            tried.add(tier.model)
            budget.block(tier, f"rate limited — {str(exc)[:60]}", persist=True)
        except llm_gemini.Overloaded as exc:
            # Busy, not broken. Bench it briefly so a LATER batch can come
            # back to it — this is the whole reason the four best models had
            # never served a single batch: one 503 retired each of them for
            # the run and everything fell to the cheap tiers.
            log("summarize", f"  batch {n}/{total}: {tier.model} busy, "
                             f"trying the next tier")
            tried.add(tier.model)
            budget.cool(tier, str(exc)[:60])
        except llm_gemini.GeminiError as exc:
            # Config errors (bad key, retired model) are permanent for this
            # model but say nothing about the others, so block and move on.
            log("summarize", f"! batch {n}/{total} on {tier.model}: "
                             f"{str(exc)[:90]}")
            tried.add(tier.model)
            budget.block(tier, "call failed")

def top_up(items: list[Item], bodies: dict[str, str]) -> list[Item]:
    """Summarise items that will PUBLISH but were never sent to the model.

    The shortlist is ranked on the pre-LLM blend, but the model's importance
    score then reorders everything: demoting a shortlisted item lets an
    unshortlisted one rise into the publish set, arriving with only an
    extractive summary. On a live run that left 17 of 40 published cards
    without a real summary, and their keyword-interest median was HIGHER
    than the summarised ones — these were not filler.

    Widening the shortlist cannot fix this, because the promotion happens
    after the shortlist is chosen. Only a second pass can.

    This used to be bounded "to the publish cap, so the worst case is two
    extra calls". That was true while the cap was 40 and false the instant it
    became 900: the gap set would have been every unsummarised publishable
    item, roughly 750 of them, or ~38 extra batches on top of a run that
    normally makes TEN CALLS IN TOTAL. It would have emptied the daily quota
    on the first run and then failed for the rest of the day. It is bounded
    explicitly now, by llm.top_up_max.
    """
    from .config import settings as _settings
    from .s7_publish import _apply_quotas

    cfg = _settings()
    if cfg["llm"]["provider"] == "none" or not api_key():
        return items

    live = [i for i in items if i.verdict is not Verdict.REJECTED]
    live.sort(key=lambda i: (i.priority.value != "critical", -i.blend))
    will_publish = _apply_quotas(live, cfg["window"])

    # Was: search each item's trace for the words "extractive" or
    # "heuristic". Rewording that log line - which happened one commit ago -
    # silently emptied this list and turned the whole top-up pass off with no
    # error anywhere. The field says what the string was standing in for.
    gaps = [i for i in will_publish if i.summary_source != "model"]
    if not gaps:
        return items

    budget = int(cfg["llm"].get("top_up_max", 40))
    if len(gaps) > budget:
        # Highest blend first: these are the items the model's own reordering
        # pushed up, so the top of that list is exactly what the pass exists
        # to rescue.
        gaps.sort(key=lambda i: -i.blend)
        log("summarize", f"top-up: {len(gaps)} publishable items lack a model "
                         f"summary, taking the top {budget}")
        gaps = gaps[:budget]
    else:
        log("summarize", f"top-up: {len(gaps)} publishable items have no model "
                         f"summary")
    # run() mutates the Item objects in place and returns the list it was
    # given, so the ORIGINAL list must be returned here — returning run()'s
    # value would silently reduce the pipeline to just the gap items.
    run(gaps, bodies, _is_top_up=True)
    return items


# --------------------------------------------------------------- main ----

def run(items: list[Item], bodies: dict[str, str],
        _is_top_up: bool = False) -> list[Item]:
    cfg = settings()["llm"]
    provider = cfg["provider"]

    if provider == "none":
        return _fallback_all(items, bodies,
                             f"provider=none — extractive fallback for {len(items)} items")

    key = api_key()
    if not key:
        return _fallback_all(
            items, bodies,
            f"! {cfg['api_key_env']} not set — falling back to extractive")

    if provider != "gemini":
        raise NotImplementedError(
            f"provider '{provider}' is not implemented. Use 'gemini' or 'none'.")

    from . import llm_gemini

    cache = _load_cache()
    profile = interests()["profile"]

    # Only a shortlist is worth paying for. Items outside it still get an
    # extractive summary so nothing is left blank, but they are far below the
    # publish cut and the model's opinion of them would never be read.
    if _is_top_up:
        shortlist, rest = items, []      # caller already chose the exact set
    else:
        # The per-category allocation used to be the PUBLISH quota. That
        # coupling broke the moment the page started publishing everything
        # in the window: per_category_max is now a guard against one wire
        # owning the page, not a page size, so reading it here asked the
        # model for hundreds of summaries the budget cannot pay for.
        # The shortlist is a spending decision, so derive it from the
        # spending limit.
        cats = len({i.category for i in items}) or 1
        per_cat = max(1, cfg["max_items_summarized"] // cats)
        shortlist, rest = _shortlist(
            items, cfg["max_items_summarized"], per_cat,
            cfg.get("shortlist_multiplier", 2.0))
    if rest:
        log("summarize", f"shortlist {len(shortlist)} to the model, "
                         f"{len(rest)} extractive (below the publish cut)")
        for item in rest:
            _heuristic(item, bodies.get(item.cluster_key, ""))

    pending: list[dict] = []
    cached = 0
    for item in shortlist:
        body = bodies.get(item.cluster_key, "")
        # Cache key includes the body so a re-extracted, fuller article gets
        # a fresh summary rather than reusing one written from a stub.
        # Try the cluster key first, then every member article's uid. The
        # cluster key moves when membership changes — a story picked up by a
        # second outlet reseeds it — so a single-key lookup missed summaries
        # already paid for. Measured hit rate was 21% against an expected
        # ~50% for a rolling window run daily.
        ckey = f"{item.cluster_key}:{len(body)}"
        hit = next((k for k in [ckey] + [f"{a}:{len(body)}" for a in item.alt_keys]
                    if k in cache), None)
        if hit:
            _apply(item, cache[hit], "cache", body)
            # Moves it to the END of the order, so the trim in _save_cache
            # evicts what is genuinely unused rather than what was written
            # longest ago, and drops the old key instead of duplicating it.
            _touch(cache, hit, ckey)
            cached += 1
            continue
        pending.append({"id": item.cluster_key, "title": item.title,
                        "source": item.sources[0]["name"] if item.sources else "",
                        "body": body, "_item": item, "_ckey": ckey})

    log("summarize", f"{cached} from cache, {len(pending)} to generate")
    if not pending:
        return items

    by_id = {p["id"]: p for p in pending}
    size = cfg["batch_size"]
    batches = [pending[i:i + size] for i in range(0, len(pending), size)]

    # Budget gate. Capacity is summed across the model ladder; if it cannot
    # cover every batch, the run is trimmed up front rather than dying
    # part-way through. Batches are already ordered best-first, so the ones
    # that get dropped are the least important.
    budget = Budget()
    # Pass the batch size so tiers that slice a batch into several calls are
    # counted as the fraction of a batch they can actually serve.
    capacity = budget.capacity(size)
    if capacity < len(batches):
        log("summarize", f"! quota covers {capacity}/{len(batches)} batches")
        log("budget", budget.report())
        for b in [x for batch in batches[capacity:] for x in batch]:
            _heuristic(b["_item"], b["body"])
            b["_item"].note("beyond API quota — extractive fallback")
        batches = batches[:capacity]

    if not batches:
        log("summarize", f"! no calls permitted — {budget.stopped_reason}")
        _save_cache(cache)
        return items

    log("summarize", f"{len(batches)} call(s), batch size {size}")
    tag_list = ", ".join(sorted(tagging.vocabulary()))
    done = failed = 0

    for n, batch in enumerate(batches, 1):
        # A batch is "bulk" when most of it is low-frequency or low-ranked
        # material. Routing it to the cheap tier keeps the flagship quota for
        # the stories at the top of the page.
        bulk = sum(_is_bulk(b["_item"]) for b in batch) > len(batch) / 2
        results, used = _generate(batch, budget, key, profile, tag_list,
                                  llm_gemini, n, len(batches),
                                  prefer="bulk" if bulk else "premium")
        if results is None:
            for b in batch:
                _heuristic(b["_item"], b["body"])
                b["_item"].note("no model could serve this batch — extractive")
                failed += 1
            continue

        seen_ids = set()
        for res in results:
            rid = res.get("id")
            src = by_id.get(rid)
            if not src:
                log("summarize", f"! unknown id in response: {rid!r}")
                continue
            seen_ids.add(rid)
            _apply(src["_item"], res, used, src["body"])
            cache[src["_ckey"]] = res
            done += 1

        # A model that silently drops an item would leave it with no summary
        # at all; catch that rather than publishing a blank card.
        for b in batch:
            if b["id"] not in seen_ids:
                _heuristic(b["_item"], b["body"])
                b["_item"].note("omitted by the model — extractive fallback")
                failed += 1

        log("summarize", f"batch {n}/{len(batches)} ok via {used} "
                         f"({len(seen_ids)} items)")

    _save_cache(cache)
    log("summarize", f"{done} generated, {cached} cached, {failed} fell back")
    log("budget", budget.report())
    return items
