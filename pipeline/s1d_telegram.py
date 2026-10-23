"""Stage 1d — public Telegram channels.

Telegram exposes a read-only web preview of any PUBLIC channel at
``t.me/s/<name>``. No bot, no token, no API key, no account — verified
against several channels before this was written. Private channels have no
such preview and would need a bot added to them; that is not supported here.

Channels are a genuinely different source class from RSS. A CTI channel
posts a link plus a line of context within minutes of something breaking,
often well before anyone writes an article about it. What they do not give
is article text, so most items arrive as a headline and a URL — stage 3
fetches the real body from the link.

Many channels post through link shorteners (ift.tt, bit.ly, buff.ly).
Those are resolved here, because an unresolved shortener breaks three things
at once: source tiering, de-duplication, and the reference link a reader
clicks.
"""

from __future__ import annotations

import html
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

from . import netguard
from .config import settings
from .models import Article
from .utils import headline, log, truncate

try:
    import certifi
    _SSL = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL = ssl.create_default_context()

PREVIEW = "https://t.me/s/{channel}"

# One regex per MESSAGE, not one for text and another for time.
#
# The previous version matched text divs and <time> tags independently and
# zipped them with `stamps[-len(blocks):]`. That is only correct when every
# extra timestamp precedes the first text block, which is not how a channel
# looks: a media-only post (routine on vxunderground) produces a stamp with
# no text block, and a REPLY produces two text blocks for one stamp because
# the reply-preview div also starts with `tgme_widget_message_text`. Either
# way every subsequent post was paired with the wrong time — silently, since
# when the counts happen to match the slice is a no-op.
_MESSAGE = re.compile(
    r'<div class="tgme_widget_message[ "].*?(?=<div class="tgme_widget_message[ "]|\Z)',
    re.S)
_TEXT_IN_MSG = re.compile(
    r'<div class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>', re.S)
_TIME_IN_MSG = re.compile(r'<time datetime="([^"]+)"')
_POST_ID = re.compile(r'data-post="[^"/]+/(\d+)"')
_HREF = re.compile(r'href="(https?://[^"]+)"')
_TAG = re.compile(r"<[^>]+>")
_BR = re.compile(r"<br\s*/?>", re.I)

# Hosts that only ever redirect. Their domain says nothing about the source.
# Shorteners chain at most a couple of times in practice; past that it is
# either a loop or someone being clever.
_MAX_HOPS = 3

SHORTENERS = {"ift.tt", "bit.ly", "buff.ly", "t.co", "dlvr.it", "ow.ly",
              "tinyurl.com", "lnkd.in", "trib.al", "shorturl.at", "rb.gy"}



def _known_sources() -> dict[str, tuple[str, str]]:
    """domain -> (display name, tier) for every outlet we already harvest.

    A channel post is a POINTER, not a source. When ctinow links to
    bleepingcomputer.com, the article is BleepingComputer's — tiering it C
    because a Telegram channel happened to surface it understates it badly,
    and on a live run left genuine security reporting sitting below the
    publication bar.
    """
    from .config import feeds
    from .utils import domain_of

    out: dict[str, tuple[str, str]] = {}
    for specs in feeds().values():
        for spec in specs:
            if spec.get("aggregator"):
                continue
            dom = domain_of(spec["url"])
            if dom:
                out[dom] = (spec["name"], spec.get("tier", "C"))
    # Outlets that show up through channels but are not feeds of ours.
    out.setdefault("securityweek.com", ("SecurityWeek", "B"))
    out.setdefault("therecord.media", ("The Record", "B"))
    out.setdefault("infosecurity-magazine.com", ("Infosecurity", "B"))
    out.setdefault("helpnetsecurity.com", ("Help Net Security", "B"))
    out.setdefault("securityaffairs.com", ("Security Affairs", "B"))
    return out


def _fetch_page(url: str, ua: str, timeout: int,
                attempts: int = 3) -> str | None:
    """One preview page, with backoff.

    t.me resets connections under concurrent load - observed repeatedly with
    WinError 10054 while probing five channels at once. Without a retry a
    single reset dropped that channel for the whole day, silently, and the
    collector reported nothing at all.
    """
    import random
    import time

    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": ua})
            with urllib.request.urlopen(req, timeout=timeout,
                                        context=_SSL) as resp:
                return resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            # 404 means no public preview: private, renamed, or nonexistent.
            # Retrying will not change that.
            if exc.code == 404:
                log("telegram", f"! {url}: HTTP 404 (private or missing?)")
                return None
            if attempt == attempts:
                log("telegram", f"! {url}: HTTP {exc.code} after {attempts} tries")
                return None
        except Exception as exc:
            if attempt == attempts:
                log("telegram", f"! {url}: {exc} (after {attempts} tries)")
                return None
        # Jittered backoff: a fixed sleep would resynchronise the workers
        # that just collided.
        # Patient on purpose. t.me rate-limits by IP, and the penalty
        # outlasts a one-second pause: repeated probing during development
        # put this machine into a state where every request was reset for
        # minutes. A daily run never sees that, but a retried run should not
        # make it worse either.
        time.sleep(attempt * 3.0 + random.random() * 2)
    return None


def _fetch(channel: str, ua: str, timeout: int, max_pages: int = 1) -> str | None:
    """The channel preview, walking back through older pages if asked.

    t.me/s/<channel> returns only the ~20 most recent messages. For a channel
    posting more than that between runs - ctinow files about 20 a day and
    came back 20-of-20 kept with its newest post 0 hours old, which is what
    saturation looks like - everything past the first page was invisible.
    ?before=<id> walks backwards, so pages are concatenated and the existing
    per-message parsing sees the lot.
    """
    pages: list[str] = []
    url = PREVIEW.format(channel=channel)
    for _ in range(max(1, max_pages)):
        page = _fetch_page(url, ua, timeout)
        if not page:
            break
        pages.append(page)
        ids = _POST_ID.findall(page)
        if not ids:
            break
        try:
            oldest = min(int(i) for i in ids)
        except ValueError:
            break
        if oldest <= 1:
            break
        url = PREVIEW.format(channel=channel) + f"?before={oldest}"
    return "".join(pages) if pages else None


def _resolve_short(url: str, ua: str, timeout: int) -> str:
    """Follow a shortener to its destination. Returns the input on failure."""
    # `hasattr(urllib.request, "urlparse")` is always true — CPython re-exports
    # it — so the fallback branch here was unreachable and the www. prefix was
    # never stripped. A `www.bit.ly` link therefore missed SHORTENERS and went
    # unresolved, breaking the three things this function exists to protect:
    # source tiering, de-duplication, and the reference link a reader clicks.
    from urllib.parse import urlsplit

    host = (urlsplit(url).hostname or "").lower().removeprefix("www.")
    if host not in SHORTENERS:
        return url
    # Check EVERY hop before following it, and never let urllib follow one on
    # its own. The old version let urlopen chase the whole redirect chain and
    # only inspected the final URL afterwards — with resolve_dns=False, so a
    # scheme check and nothing more. By then the request had already been
    # made: a shortener pointing at http://169.254.169.254/ (or any host on
    # the LAN) was fetched first and vetted second, which is the wrong order
    # for the one thing netguard exists to prevent. The surviving link was
    # then published as the card's destination.
    #
    # Unlike stage 3, there is no library guard underneath this — trafilatura
    # vets peers post-connect on every hop, urllib does not — so the check has
    # to be here.
    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None            # stop; the caller inspects Location itself

    opener = urllib.request.build_opener(_NoRedirect)
    seen = 0
    while seen <= _MAX_HOPS:
        reason = netguard.check(url)
        if reason:
            log("telegram", f"! refusing redirect to {url[:48]}: {reason}")
            return ""
        try:
            req = urllib.request.Request(url, headers={"User-Agent": ua},
                                         method="HEAD")
            with opener.open(req, timeout=timeout) as resp:
                return url                      # 2xx: this is the destination
        except urllib.error.HTTPError as exc:
            nxt = exc.headers.get("Location") if 300 <= exc.code < 400 else None
            if not nxt:
                return url
            url = urllib.parse.urljoin(url, nxt)
            seen += 1
        except Exception:
            return url
    return url


def _parse(block: str) -> tuple[str, str | None]:
    """(plain text, first external link) from one message's HTML."""
    text = html.unescape(_TAG.sub("", _BR.sub("\n", block))).strip()
    links = [u for u in _HREF.findall(block) if "t.me" not in u]
    return " ".join(text.split()), (links[0] if links else None)


def _channel_articles(spec: dict, cutoff: datetime, cfg: dict) -> list[Article]:
    ua = cfg["harvest"]["user_agent"]
    timeout = cfg["harvest"]["timeout_seconds"]
    channel = spec["channel"]
    tg = cfg["telegram"]

    # A channel that fills its first page every run is truncated, so let the
    # busy ones walk back further. Per-channel because ctinow needs several
    # pages a day and secharvester needs one a week.
    pages = int(spec.get("max_pages") or tg.get("max_pages", 1))
    page = _fetch(channel, ua, timeout, max_pages=pages)
    if not page:
        return []

    # A channel's own window, when it has one. Channels post at wildly
    # different rates: ctinow runs 20 posts a day, secharvester about one a
    # week. Judging both against a single 72h cutoff does not filter the slow
    # ones, it DELETES them - every post is always older than the window, so
    # the channel contributes nothing on every run for ever. feeds.yaml has
    # had per-feed windows for exactly this reason; channels need them too.
    own = spec.get("window_hours")
    if own:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=int(own))

    out: list[Article] = []
    # Counted so a channel that yields nothing can say WHY. Silence here was
    # the whole problem: three of five channels returned zero and logged not
    # one line, so a dead channel, a slow channel and a channel whose posts
    # were all filtered out looked exactly alike from the console.
    drop = {"short": 0, "old": 0, "nolink": 0, "unsafe": 0}

    # Each message carries its own timestamp, so they cannot drift apart.
    for chunk in _MESSAGE.findall(page):
        texts = _TEXT_IN_MSG.findall(chunk)
        stamps = _TIME_IN_MSG.findall(chunk)
        if not texts or not stamps:
            continue                      # media-only post, or no timestamp

        # The LAST text div is the message's own; an earlier one is the
        # quoted post in a reply. The LAST <time> is the post time; an
        # earlier one belongs to the quoted post.
        block, stamp = texts[-1], stamps[-1]
        text, link = _parse(block)
        if len(text) < tg["min_chars"]:
            drop["short"] += 1
            continue
        try:
            when = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        except ValueError:
            continue
        if when < cutoff:
            drop["old"] += 1
            continue

        # A post with no link is commentary. Useful to a human scrolling the
        # channel, but there is nothing for the pipeline to verify against.
        if not link and tg.get("require_link", True):
            drop["nolink"] += 1
            continue

        url = _resolve_short(link, ua, timeout) if link else \
            f"https://t.me/{channel}"
        if not netguard.is_safe(url, resolve_dns=False):
            drop["unsafe"] += 1
            continue

        # Strip the trailing URL from the title — it is already the link.
        title = re.sub(r"https?://\S+$", "", text).strip() or text
        from .utils import domain_of

        dom = domain_of(url)
        # Credit the outlet the link actually points at, not the channel that
        # surfaced it. A post pointing at BleepingComputer IS BleepingComputer.
        known = _known_sources().get(dom)
        if known:
            src_name, src_tier = known
        else:
            src_name = spec.get("name", channel)
            src_tier = spec.get("tier", "C")   # unknown host: one poster's word

        out.append(Article(
            url=url,
            title=headline(title, 160),
            source=src_name,
            domain=dom,
            category=spec["category"],
            tier=src_tier,
            published=when,
            summary_raw=text,
            is_primary=False,
            window_hours=int(spec.get("window_hours") or 0),
        ))

    # One line per channel, always. A zero with its reason attached is
    # actionable ("raise this channel's window"); a zero with no line at all
    # is indistinguishable from the collector never having run.
    if out:
        log("telegram", f"  {channel:<24} {len(out):>2} posts")
    else:
        why = ", ".join(f"{v} {k}" for k, v in drop.items() if v) or "no messages"
        window = int(own) if own else tg["lookback_hours"]
        log("telegram", f"  {channel:<24}  0 posts  ({why}; window {window}h)")
    return out


def run() -> list[Article]:
    cfg = settings()
    tg = cfg.get("telegram")
    if not tg or not tg.get("enabled", False):
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(hours=tg["lookback_hours"])
    channels = tg.get("channels", [])

    out: list[Article] = []
    # Two, not four. t.me resets connections under concurrent load: at four
    # workers three of five channels died with WinError 10054 on a single
    # run, and pagination multiplies the request count per channel. The
    # collector is not on the critical path for time - the whole stage is
    # seconds - so trade throughput for actually getting the posts.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_channel_articles, c, cutoff, cfg)
                   for c in channels]
        for fut in as_completed(futures):
            try:
                out.extend(fut.result())
            except Exception as exc:
                log("telegram", f"! channel failed: {exc}")

    log("telegram", f"{len(out)} posts from {len(channels)} channel(s)")
    return out
