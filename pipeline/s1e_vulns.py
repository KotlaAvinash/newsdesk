"""Stage 1e — vulnerability intelligence from JSON APIs.

Two of the most authoritative sources for this dashboard publish no usable
feed, and both were previously written off:

  * **CISA KEV** — the Known Exploited Vulnerabilities catalogue. The XML
    endpoint returns 403; the JSON one returns 200. This is the single best
    signal the pipeline can carry, because inclusion means CISA has evidence
    of exploitation IN THE WILD, not merely that a CVE exists.
  * **NVD** — NIST retired its RSS feeds entirely. The 2.0 REST API is free,
    needs no key, and serves the same data.

Both are primary artifacts, so items from here are marked `is_primary` and
anchor a cluster to `verified` on their own.

Rate limits: NVD asks for 5 requests per 30s without a key (50 with one).
This makes ONE request per run, so that is not a concern — but do not raise
`results_per_page` and start paginating without adding sleeps.
"""

from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

from .config import settings
from .models import Article
from .utils import log, truncate

try:
    import certifi
    _SSL = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL = ssl.create_default_context()

KEV_URL = ("https://www.cisa.gov/sites/default/files/feeds/"
           "known_exploited_vulnerabilities.json")
NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"


def _get(url: str, cfg: dict) -> bytes | None:
    req = urllib.request.Request(
        url, headers={"User-Agent": cfg["harvest"]["user_agent"]})
    try:
        with urllib.request.urlopen(
                req, timeout=cfg["harvest"]["timeout_seconds"], context=_SSL) as r:
            return r.read()
    except Exception as exc:
        log("vulns", f"! {url.split('/')[2]}: {exc}")
        return None


def _kev(cfg: dict, cutoff: datetime) -> list[Article]:
    """Newly catalogued exploited vulnerabilities."""
    raw = _get(KEV_URL, cfg)
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        log("vulns", "! KEV returned non-JSON")
        return []

    out: list[Article] = []
    for v in data.get("vulnerabilities", []):
        try:
            added = datetime.fromisoformat(v["dateAdded"]).replace(
                tzinfo=timezone.utc)
        except (KeyError, ValueError):
            continue
        if added < cutoff:
            continue

        cve = v.get("cveID", "")
        vendor = v.get("vendorProject", "")
        product = v.get("shortDescription", "")
        action = v.get("requiredAction", "")
        due = v.get("dueDate", "")
        ransom = v.get("knownRansomwareCampaignUse", "")

        body = (f"{v.get('vulnerabilityName', '')}. {product}\n\n"
                f"Required action: {action}\n"
                f"Federal remediation due: {due}\n"
                f"Known ransomware campaign use: {ransom}\n"
                f"Added to the CISA KEV catalogue on {v['dateAdded']}.")

        out.append(Article(
            url=f"https://nvd.nist.gov/vuln/detail/{cve}",
            title=f"{cve}: {vendor} {v.get('vulnerabilityName', '')}".strip(),
            source="CISA KEV",
            domain="cisa.gov",
            category="cyber_attacks",
            tier="A",
            published=added,
            summary_raw=body,
            # This record IS the artifact — there is no article behind the
            # link to fetch. Measured 2026-10-22: all four cluster leads
            # pointing at nvd.nist.gov/vuln/detail/ returned the same
            # 2,453-byte "enable JavaScript" shell and extracted to nothing,
            # while the KEV JSON above had already supplied a 937-1,013
            # character description. Setting body here means stage 3 skips
            # the fetch instead of making it and discarding the result.
            body=body,
            is_primary=True,     # inclusion means confirmed exploitation
        ))
    if out:
        log("vulns", f"  CISA KEV               {len(out):>3} newly catalogued")
    return out


def _nvd(cfg: dict, cutoff: datetime, cvss_floor: float,
         limit: int) -> list[Article]:
    """Recently published CVEs at or above a severity floor."""
    params = {
        "pubStartDate": cutoff.strftime("%Y-%m-%dT%H:%M:%S.000"),
        "pubEndDate": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000"),
        "resultsPerPage": str(limit),
        "noRejected": "",
    }
    raw = _get(f"{NVD_URL}?{urllib.parse.urlencode(params)}", cfg)
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        log("vulns", "! NVD returned non-JSON")
        return []

    out: list[Article] = []
    for entry in data.get("vulnerabilities", []):
        cve = entry.get("cve", {})
        cve_id = cve.get("id", "")

        # Severity lives under several possible metric versions.
        score, severity = 0.0, ""
        metrics = cve.get("metrics", {})
        for key in ("cvssMetricV40", "cvssMetricV31", "cvssMetricV30"):
            if metrics.get(key):
                d = metrics[key][0].get("cvssData", {})
                score = d.get("baseScore", 0.0)
                severity = d.get("baseSeverity", "")
                break
        if score < cvss_floor:
            continue

        desc = next((d["value"] for d in cve.get("descriptions", [])
                     if d.get("lang") == "en"), "")
        try:
            published = datetime.fromisoformat(
                cve["published"]).replace(tzinfo=timezone.utc)
        except (KeyError, ValueError):
            continue

        record = (f"{desc}\n\nCVSS base score {score} ({severity}). "
                  f"Published {cve['published'][:10]}.")
        out.append(Article(
            url=f"https://nvd.nist.gov/vuln/detail/{cve_id}",
            title=f"{cve_id} ({severity} {score}): {truncate(desc, 110)}",
            source="NVD",
            domain="nist.gov",
            category="cyber_attacks",
            tier="A",
            published=published,
            summary_raw=record,
            # Same as KEV above: the API response is the artifact and the
            # detail page is a JavaScript shell, so stage 3 has nothing to
            # gain by fetching it.
            body=record,
            is_primary=True,
        ))
    if out:
        log("vulns", f"  NVD (CVSS>={cvss_floor})       {len(out):>3} vulnerabilities")
    return out


def run() -> list[Article]:
    cfg = settings()
    vc = cfg.get("vulns")
    if not vc or not vc.get("enabled", False):
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(hours=vc["lookback_hours"])
    out: list[Article] = []

    if vc.get("kev", True):
        out += _kev(cfg, cutoff)
    if vc.get("nvd", True):
        out += _nvd(cfg, cutoff, vc.get("nvd_cvss_floor", 8.0),
                    vc.get("nvd_limit", 60))

    log("vulns", f"{len(out)} vulnerability records")
    return out
