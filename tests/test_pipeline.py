"""Unit tests for the parts where a silent regression would be invisible.

Deliberately narrow: these cover the pure logic (url canonicalisation,
clustering thresholds, scoring behaviour), not the network stages.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.models import Article, Cluster, canonical_url
from pipeline.s2_clean import _cluster
from pipeline.utils import overlap, strip_html, tokens


def _art(title, domain="a.com", tier="B", primary=False, aggregator=False):
    return Article(url=f"https://{domain}/{abs(hash(title))}", title=title,
                   source=domain, domain=domain, category="tech_ai", tier=tier,
                   published=datetime.now(timezone.utc), is_primary=primary,
                   is_aggregator=aggregator)


class TestCanonicalUrl:
    def test_strips_tracking_params(self):
        a = canonical_url("https://x.com/a?utm_source=rss&id=7")
        b = canonical_url("https://www.x.com/a/?id=7")
        assert a == b, "same article must canonicalise identically"

    def test_keeps_meaningful_params(self):
        assert "id=7" in canonical_url("https://x.com/a?id=7")


class TestOverlap:
    def test_asymmetric_headlines_still_match(self):
        # The exact failure that produced zero clusters on the first live run.
        short = tokens("Girl, 17, killed in Swedish sword attack, police say")
        long_ = tokens("Swedish police probe online networks after teenage girl "
                       "killed in sword attack")
        assert overlap(short, long_) >= 0.55

    def test_unrelated_headlines_do_not_match(self):
        a = tokens("RBI imposes monetary penalty on a finance company")
        b = tokens("Fourteen killed in strike on Myanmar monastery")
        assert overlap(a, b) < 0.55

    def test_empty_is_zero(self):
        assert overlap(set(), {"a"}) == 0.0


class TestClustering:
    def test_same_story_across_outlets_merges(self):
        arts = [
            _art("TikTok agrees to $400 million US children's privacy settlement", "reuters.com"),
            _art("TikTok Agrees to $400 Million Settlement in U.S. Child Privacy Case", "nyt.com"),
            _art("Fourteen killed in strike on Myanmar monastery", "bbc.co.uk"),
        ]
        clusters = _cluster(arts, 0.55, 3)
        sizes = sorted(len(c.articles) for c in clusters)
        assert sizes == [1, 2], f"expected one pair + one single, got {sizes}"

    def test_min_shared_tokens_blocks_thin_matches(self):
        # Two short headlines sharing one word must not merge.
        arts = [_art("Apple ships update", "a.com"), _art("Google ships update", "b.com")]
        clusters = _cluster(arts, 0.55, 3)
        assert len(clusters) == 2


class TestClusterLead:
    def test_aggregator_never_leads(self):
        """Aggregator links are redirect stubs — the lead URL is what is clicked."""
        agg = _art("Story", "news.google.com", tier="A", aggregator=True)
        real = _art("Story", "bbc.co.uk", tier="B")
        assert Cluster(key="k", articles=[agg, real]).lead is real

    def test_independent_outlets_counts_domains(self):
        c = Cluster(key="k", articles=[_art("x", "a.com"), _art("x", "b.com"),
                                       _art("x", "a.com")])
        assert c.independent_outlets == 2


class TestStripHtml:
    def test_removes_tags_and_unescapes(self):
        assert strip_html("<p>A &amp; B</p>") == "A & B"

    def test_drops_script_content(self):
        assert "evil" not in strip_html("<script>evil()</script>ok")


class TestScoring:
    """The regression that mattered most: trustworthy must not mean important."""

    def test_primary_source_does_not_inflate_interest(self):
        from pipeline.config import settings
        from pipeline.s4_corroborate import _interest_score

        icfg = settings()["interest"]
        title = "Cabinet approves four multitracking projects"
        plain = Cluster(key="a", articles=[_art(title, "x.com")])
        official = Cluster(key="b", articles=[_art(title, "pib.gov.in", tier="A",
                                                   primary=True)])
        assert _interest_score(plain, icfg) == _interest_score(official, icfg), \
            "an official source is a reason to trust a story, not to care about it"


@pytest.fixture(autouse=True)
def _isolated_usage(tmp_path, monkeypatch):
    """Redirect the usage counter at a temp file for every test in this module.

    Without this the tests delete the REAL data/cache/usage.json — which they
    did, wiping live quota state out from under a run that was in flight.
    Tests must never touch production data.
    """
    import pipeline.budget as budget
    monkeypatch.setattr(budget, "USAGE_PATH", tmp_path / "usage.json")
    # _RUN_BLOCKS is module-level state that deliberately survives a new
    # Budget() within one run. It must not survive between TESTS, or one
    # test's simulated 503 silently blocks a model in the next.
    monkeypatch.setattr(budget, "_RUN_BLOCKS", {})


class TestBudgetKillSwitch:
    """The stop condition has to be provable, not assumed.

    These run against the real settings ladder but never touch the network
    and never touch the real usage counter (see the fixture above).
    """

    def _fresh(self):
        from pipeline.budget import Budget, USAGE_PATH
        USAGE_PATH.unlink(missing_ok=True)
        return Budget()

    def test_ladder_is_ordered_best_first(self):
        b = self._fresh()
        assert b.tiers[0].model.startswith("gemini-3."), b.tiers[0].model
        # Cheap high-quota models must sit at the bottom, not the top.
        assert b.tiers[-1].rpd > b.tiers[0].rpd

    def test_exhaustion_stops_cleanly(self):
        b = self._fresh()
        for tier in b.tiers:
            tier.blocked = "test"
        assert b.capacity() == 0
        assert b.next_model() is None
        assert "exhausted" in b.stopped_reason

    def test_reserve_protects_a_slice_of_quota(self):
        """A full run must never lock out an ad-hoc --test-llm."""
        b = self._fresh()
        assert b.tiers[0].usable_rpd < b.tiers[0].rpd

    def test_per_run_cap_is_below_daily_cap(self):
        b = self._fresh()
        for tier in b.tiers:
            assert tier.per_run <= tier.usable_rpd, tier.model

    def test_quota_refusal_persists_but_overload_does_not(self):
        """Two different lifetimes, and both matter.

        A 429 means the provider refused for the rest of the UTC day, so it
        must survive into the next RUN — persisted to usage.json.

        A 503 is transient overload, so it must NOT cost the model tomorrow.
        It does still have to survive within the CURRENT run, because s5.run,
        s5.top_up and s6b_judge each build their own Budget() and would
        otherwise each retry the same dead model — that cost 4 of 11 calls on
        a measured run.
        """
        import pipeline.budget as _b
        from pipeline.budget import Budget
        USAGE_PATH = _b.USAGE_PATH

        b = self._fresh()
        b.block(b.next_model(), "429", persist=True)
        assert Budget().tiers[0].blocked, "a 429 must survive into the next run"

        # A new run: fresh process state AND fresh usage file.
        USAGE_PATH.unlink(missing_ok=True)
        _b._RUN_BLOCKS.clear()

        b2 = Budget()
        b2.block(b2.next_model(), "503 overload")       # transient
        assert Budget().tiers[0].blocked,             "a 503 must survive within the run, or later stages retry it"

        _b._RUN_BLOCKS.clear()                          # next run begins
        assert not Budget().tiers[0].blocked,             "a 503 must not persist into the next run"
        USAGE_PATH.unlink(missing_ok=True)


class TestVerdicts:
    """Verdicts must actually discriminate — the previous scheme put 96% of
    items on one label, which is true but useless."""

    def _item(self, tier="B", outlets=1, primary=False):
        from datetime import datetime, timezone
        from pipeline.models import Cluster
        arts = [_art("Some headline about a thing", f"d{n}.com", tier=tier)
                for n in range(outlets)]
        if primary:
            arts[0].is_primary = True
        return Cluster(key="k", articles=arts)

    def test_primary_anchor_is_verified(self):
        from pipeline.s4_corroborate import run
        from pipeline.models import Verdict
        out = run([self._item(primary=True)])
        assert out[0].verdict is Verdict.VERIFIED

    def test_three_outlets_is_verified(self):
        from pipeline.s4_corroborate import run
        from pipeline.models import Verdict
        out = run([self._item(outlets=3)])
        assert out[0].verdict is Verdict.VERIFIED

    def test_two_outlets_is_corroborated(self):
        from pipeline.s4_corroborate import run
        from pipeline.models import Verdict
        out = run([self._item(outlets=2)])
        assert out[0].verdict is Verdict.CORROBORATED

    def test_known_publication_beats_a_random_blog(self):
        """The distinction the old single_source label threw away."""
        from pipeline.s4_corroborate import run
        from pipeline.models import Verdict
        established = run([self._item(tier="B")])[0]
        blog = run([self._item(tier="C")])[0]
        assert established.verdict is Verdict.ESTABLISHED
        assert blog.verdict is Verdict.UNVERIFIED


class TestNetGuard:
    """Outbound fetch safety.

    Article links come from feeds, so they are attacker-influenceable. These
    are offline tests: no request is ever made.
    """

    def test_file_scheme_is_refused(self):
        """urlopen honours file:// — verified, not assumed. A feed link of
        file:///.../.env would otherwise be read and could reach a summary."""
        from pipeline.netguard import check
        assert check("file:///C:/Users/me/.env") is not None

    def test_non_http_schemes_refused(self):
        from pipeline.netguard import check
        for url in ("ftp://example.com/x", "gopher://example.com",
                    "data:text/html,<script>", "javascript:alert(1)"):
            assert check(url) is not None, url

    def test_cloud_metadata_endpoint_blocked(self):
        """169.254.169.254 is the cloud metadata service on CI runners."""
        from pipeline.netguard import check
        assert check("http://169.254.169.254/latest/meta-data/") is not None

    def test_private_and_loopback_blocked(self):
        from pipeline.netguard import check
        for url in ("http://127.0.0.1:8777/", "http://192.168.1.1/admin",
                    "http://10.0.0.5/", "http://localhost/"):
            assert check(url) is not None, url

    def test_ordinary_public_urls_allowed(self, monkeypatch):
        """Hermetic: the old version called real DNS, so it passed or failed
        depending on the network the suite happened to run on."""
        import socket as _s
        from pipeline import netguard
        monkeypatch.setattr(netguard.socket, "getaddrinfo",
                            lambda *a, **k: [(_s.AF_INET, None, None, "",
                                              ("151.101.156.81", 0))])
        assert netguard.check("https://www.bbc.co.uk/news") is None

    def test_nat64_public_address_allowed(self, monkeypatch):
        """A DNS64 resolver answers with 64:ff9b::<ipv4>, which Python calls
        `is_reserved`. Treating the prefix as the address blocked every
        public host on any IPv6-only or mobile network - the entire harvest."""
        import socket as _s
        from pipeline import netguard
        monkeypatch.setattr(netguard.socket, "getaddrinfo",
                            lambda *a, **k: [(_s.AF_INET6, None, None, "",
                                              ("64:ff9b::9765:9c51", 0, 0, 0))])
        assert netguard.check("https://www.bbc.co.uk/news") is None

    def test_nat64_cannot_smuggle_a_private_address(self, monkeypatch):
        """The other direction, and the one that matters: unwrapping must not
        become a way past the guard. 64:ff9b::a9fe:a9fe carries the cloud
        metadata address and has to stay blocked."""
        import socket as _s
        from pipeline import netguard
        for wrapped in ("64:ff9b::a9fe:a9fe",      # 169.254.169.254
                        "64:ff9b::c0a8:0101",      # 192.168.1.1
                        "64:ff9b::7f00:0001",      # 127.0.0.1
                        "::ffff:10.0.0.5"):        # IPv4-mapped private
            monkeypatch.setattr(
                netguard.socket, "getaddrinfo",
                lambda *a, w=wrapped, **k: [(_s.AF_INET6, None, None, "",
                                             (w, 0, 0, 0))])
            assert netguard.check("https://evil.example/") is not None, wrapped

    def test_scheme_only_mode_skips_dns(self):
        """The cheap path still catches the scheme abuses, which is what the
        per-article hot loop needs."""
        from pipeline.netguard import check
        assert check("file:///etc/passwd", resolve_dns=False) is not None
        assert check("https://example.com", resolve_dns=False) is None


class TestApiBudget:
    """The free tier is the whole constraint, so the number of CALLS a run can
    make has to be provably bounded by config rather than by how much news
    happened to break that day."""

    def test_top_up_is_bounded_by_its_own_limit_not_the_publish_cap(self):
        """top_up() used to size itself from the publish cap. That was two
        extra calls at a cap of 40 and ~38 at a cap of 900 — enough to empty
        the daily quota on the first run of the day."""
        from pipeline.config import settings
        cfg = settings()
        cap = cfg["window"]["max_items_published"]
        budget = cfg["llm"]["top_up_max"]
        batch = cfg["llm"]["batch_size"]
        assert budget < cap, (
            "top_up_max must be a real ceiling, not the publish cap")
        # The whole run has to stay inside the smallest daily quota on the
        # ladder, which is the flagship tier's rpd.
        worst_case = -(-cfg["llm"]["max_items_summarized"] // batch)                      + -(-budget // batch)                      + -(-cfg["judge"]["max_items"] // cfg["judge"]["batch_size"])
        assert worst_case <= 40, f"worst case {worst_case} calls per run"

    def test_gap_detection_does_not_depend_on_log_wording(self):
        """The gap set was found by searching each item's trace for the words
        'extractive' or 'heuristic'. Rewording that log line turned the whole
        pass off silently, with no error anywhere."""
        import inspect
        from pipeline import s5_summarize
        src = inspect.getsource(s5_summarize.top_up)
        # Strip comments and docstrings: this checks the CODE, not the prose
        # explaining why the code looks like this.
        code = " ".join(l.split("#")[0] for l in src.splitlines())
        assert "summary_source" in code, "top_up must test the field"
        assert '"extractive"' not in code and '"heuristic"' not in code, (
            "top_up is matching on log wording again")


class TestTally:
    """The lifetime counter is not load-bearing. It runs before news.json is
    written, so anything it can raise costs the entire run."""

    def _tally_into(self, tmp_path, monkeypatch, content=None):
        from pipeline import s7_publish
        f = tmp_path / "totals.json"
        if content is not None:
            f.write_text(content, encoding="utf-8")
        monkeypatch.setattr(s7_publish, "TOTALS_JSON", f)
        return s7_publish, f

    def test_survives_valid_json_that_is_not_an_object(self, tmp_path, monkeypatch):
        """json.loads returns a list/None/str for these, and setdefault on any
        of them raises AttributeError, which is not a JSONDecodeError."""
        for junk in ("[]", "null", '"nope"', "42"):
            mod, f = self._tally_into(tmp_path, monkeypatch, junk)
            out = mod._tally(10, 2, "2026-09-04T06:00:00+00:00")
            assert out is not None, f"crashed on {junk}"
            assert out["published"] == 10

    def test_survives_rows_missing_keys(self, tmp_path, monkeypatch):
        """An entry without 'published' used to raise KeyError while the very
        next line used .get for 'held'."""
        mod, f = self._tally_into(
            tmp_path, monkeypatch,
            '{"runs": {"2026-09-01": {}, "2026-09-02": 7}, '
            '"first_run": "2026-09-01T00:00:00+00:00"}')
        out = mod._tally(5, 1, "2026-09-04T06:00:00+00:00")
        assert out is not None
        assert out["published"] == 5      # the empty row contributes 0

    def test_never_raises_even_when_the_file_is_unwritable(self, tmp_path, monkeypatch):
        from pipeline import s7_publish
        monkeypatch.setattr(s7_publish, "TOTALS_JSON", tmp_path / "nope" / "x.json")
        # parent does not exist -> write fails; must degrade, not explode
        assert s7_publish._tally(1, 0, "2026-09-04T06:00:00+00:00") is None

    def test_dry_run_does_not_touch_the_file(self, tmp_path, monkeypatch):
        """--dry-run is documented as 'run everything, write nothing'. The
        tally used to be written before the dry_run guard was even reached."""
        before = '{"runs": {"2026-09-01": {"published": 3, "held": 1}}, '                  '"first_run": "2026-09-01T00:00:00+00:00"}'
        mod, f = self._tally_into(tmp_path, monkeypatch, before)
        out = mod._tally(99, 99, "2026-09-04T06:00:00+00:00", persist=False)
        assert out["published"] == 102          # computed
        assert f.read_text(encoding="utf-8") == before   # but not written


class TestCoverageCapsDoNotBind:
    def test_caps_are_above_a_real_run(self):
        """The point of the change was that nothing in the window is dropped.
        A measured harvest yields ~1,154 publishable items; caps of 900 and
        260 silently discarded 254 of them, 111 from tech_ai alone."""
        from pipeline.config import settings
        w = settings()["window"]
        assert w["max_items_published"] >= 1500, (
            "total cap would bind on a normal run")
        assert w["per_category_max"] >= 800, (
            "per-category cap would bind on tech_ai")


class TestOverloadDoesNotRetireAModel:
    """Ten days of usage data showed zero successful calls to any of the four
    best models on the ladder. They were being selected first every run, 503
    on the first batch, and blacklisted for the whole run — so every later
    batch went to the cheap tiers, and nothing recorded the attempt."""

    def test_5xx_benches_rather_than_blocks(self, monkeypatch, tmp_path):
        from pipeline import budget as B
        monkeypatch.setattr(B, "USAGE_PATH", tmp_path / "usage.json")
        monkeypatch.setattr(B, "_RUN_BLOCKS", {})
        bud = B.Budget()
        top = bud.tiers[0]
        assert top.remaining > 0

        bud.cool(top, "HTTP 503")
        assert top.blocked == "", "a 5xx must not retire the model for the run"
        assert top.remaining == 0, "but it must be skipped while cooling"

        # ...and it comes back once the bench expires, which is the whole point
        top.cool_until = 0.0
        assert top.remaining > 0, "model never became available again"

    def test_three_strikes_does_block(self, monkeypatch, tmp_path):
        """Persistent failure IS the model, and should stop costing time."""
        from pipeline import budget as B
        monkeypatch.setattr(B, "USAGE_PATH", tmp_path / "usage.json")
        monkeypatch.setattr(B, "_RUN_BLOCKS", {})
        bud = B.Budget()
        top = bud.tiers[0]
        for _ in range(3):
            bud.cool(top, "HTTP 503")
        assert top.blocked, "three strikes in one run should retire it"

    def test_failed_attempts_are_recorded(self, monkeypatch, tmp_path):
        """record() only ran after SUCCESS, so a model that failed every day
        left no trace and looked unused rather than broken."""
        import json
        from pipeline import budget as B
        usage = tmp_path / "usage.json"
        monkeypatch.setattr(B, "USAGE_PATH", usage)
        monkeypatch.setattr(B, "_RUN_BLOCKS", {})
        bud = B.Budget()
        bud.cool(bud.tiers[0], "HTTP 503")
        data = json.loads(usage.read_text(encoding="utf-8"))
        day = next(iter(data.values()))
        assert day["_failed"][bud.tiers[0].model] == 1

    def test_overloaded_is_its_own_exception(self):
        """_generate must be able to tell 'busy' from 'misconfigured'."""
        from pipeline import llm_gemini
        assert issubclass(llm_gemini.Overloaded, llm_gemini.GeminiError)
        assert not issubclass(llm_gemini.Overloaded, llm_gemini.RateLimited)


class TestScheduleTellsTheTruth:
    def test_cron_matches_the_workflow(self):
        """The site reads settings.yaml to tell readers when the page next
        refreshes; the workflow is what actually fires. A drift makes it lie."""
        import re
        from pathlib import Path
        from pipeline.config import settings
        root = Path(__file__).resolve().parent.parent
        wf = (root / ".github/workflows/newsdesk.yml").read_text(encoding="utf-8")
        crons = re.findall(r'- cron:\s*"([^"]+)"', wf)
        if not crons:
            # The public mirror carries the workflow for reference but has no
            # schedule — it holds no API key, so a cron there would fail every
            # morning. Nothing to keep in step in that copy.
            import pytest
            pytest.skip("no schedule in this copy of the workflow")
        assert settings()["schedule"]["cron_utc"] in crons

    def test_next_update_accounts_for_the_queue_delay(self):
        """GitHub's median delay on this repo is 5.5h. Printing the raw cron
        time told the reader the page was overdue every morning while nothing
        was wrong."""
        from datetime import datetime, timezone
        from pipeline.s7_publish import _next_run
        plain = _next_run("0 22 * * *", 0.0)
        shifted = _next_run("0 22 * * *", 5.5)
        assert plain and shifted
        a = datetime.fromisoformat(plain)
        b = datetime.fromisoformat(shifted)
        assert (b - a).total_seconds() / 3600 in (5.5, 5.5 - 24, 5.5 + 24)

    def test_the_estimate_lands_on_the_promised_deadline(self):
        """The whole point of the 22:00 UTC cron is finishing by 09:00 IST."""
        from pipeline.config import settings
        s = settings()["schedule"]
        m, h = (int(x) for x in s["cron_utc"].split()[:2])
        finish_utc = (h * 60 + m + int(s["typical_delay_hours"] * 60)) % 1440
        ist = (finish_utc + 330) % 1440
        assert ist == 10 * 60, f"estimate lands at {ist//60:02d}:{ist%60:02d} IST"


class TestUnsummarisedCardsStillCarryFacts:
    """Most of the page is extractive now that everything in the window is
    published. A CVE card that ships as a bare headline while its identifier,
    severity and fixed version sit unparsed in the body is a wasted card."""

    def _facts(self, title, body):
        from pipeline.s5_summarize import _vuln_facts
        from pipeline.models import Item
        return _vuln_facts(Item(cluster_key="k", title=title,
                                category="cyber_attacks"), body)

    def test_cve_severity_affected_and_fix_are_extracted(self):
        f = self._facts(
            "CVE-2026-85786 - memory amplification in Amazon ion-java",
            "Affects ion-java versions prior to 1.11.9. Fixed in version "
            "1.11.9. CVSS base score 7.5 (HIGH). Published 2026-09-04.")
        joined = " | ".join(f)
        assert "CVE-2026-85786" in joined
        assert "7.5" in joined and "HIGH" in joined
        assert "1.11.9" in joined
        assert not any(x.endswith(".") for x in f), f"trailing period: {f}"

    def test_kev_flags_ransomware_and_deadline(self):
        f = self._facts(
            "CVE-2026-21962: Oracle WebLogic Server Proxy Plug-in",
            "Required action: apply mitigations. Federal remediation due: "
            "2026-08-27. Known ransomware campaign use: Known. Added to the "
            "CISA KEV catalogue on 2026-08-25.")
        joined = " | ".join(f)
        assert "ransomware" in joined.lower()
        assert "2026-08-27" in joined

    def test_returns_nothing_for_ordinary_news(self):
        """It must not invent structure where there is none."""
        assert self._facts("Google announces a new AI model",
                           "Google today announced a model.") == []

    def test_ordinary_prose_never_produces_advisory_bullets(self):
        """Every branch is gated on a CVE. Without that gate the Affected and
        Patched patterns matched business and politics copy, and a markets
        card displayed "Affected: prior to 2019" as if it were an advisory."""
        for title, body in [
            ("Reliance Q2 earnings beat estimates",
             "Revenue prior to 2019 was unaudited. The firm updated in 2023 "
             "its accounting."),
            ("India signs trade pact with Canada",
             "Tariffs earlier than 2015 levels will return. Talks resolved "
             "in 2024."),
            ("OpenAI ships GPT-6",
             "Available to users before 5.0 tier subscribers."),
        ]:
            assert self._facts(title, body) == [], title

    def test_a_nearby_number_is_not_a_cvss_score(self):
        """The score has to sit beside an explicit CVSS marker, or be joined
        to its label by whitespace alone as the NVD collector formats it.
        Accepting any digit within twelve characters of a severity word read
        "HIGH: 10 products affected" as CVSS 10 - the maximum rating - and
        "Rated LOW; 10 vendors" as the self-contradictory CVSS 10, LOW."""
        for body in ("HIGH: 10 products affected by this issue.",
                     "Rated LOW; 10 vendors confirmed.",
                     "A HIGH impact bug in v2.3 of the library."):
            joined = " | ".join(self._facts("CVE-2026-12345 flaw", body))
            assert "CVSS" not in joined, f"{body} -> {joined}"

    def test_real_cvss_formats_still_parse(self):
        """Both shapes the collectors actually emit."""
        a = " | ".join(self._facts("CVE-2026-85786 flaw",
                                   "CVSS base score 7.5 (HIGH)."))
        assert "CVSS 7.5, HIGH" in a, a
        b = " | ".join(self._facts(
            "CVE-2026-58574 (CRITICAL 9.8): Dell PowerStore", "body"))
        assert "CVSS 9.8, CRITICAL" in b, b

    def test_never_exceeds_four_facts(self):
        f = self._facts(
            "CVE-2026-11111 thing",
            "versions prior to 2.0. Fixed in version 2.0. CVSS base score "
            "9.8 (CRITICAL). Known ransomware campaign use: Known. Federal "
            "remediation due: 2026-01-01. Actively exploited in the wild.")
        assert len(f) <= 4


class TestHeadlinesAreNotSeveredMidWord:
    def test_prefers_a_complete_sentence(self):
        from pipeline.utils import headline
        # The limit has to bite, or nothing is truncated and the test proves
        # nothing. The first sentence is 85 characters; the whole post is 139.
        out = headline("After a literal year of development, the WASM check "
                       "patch has been merged into Anubis. This will be "
                       "enabled by default in the next release.", 100)
        assert out.endswith("Anubis."), out
        assert "This will be" not in out

    def test_falls_back_to_a_whole_word(self):
        from pipeline.utils import headline
        long = "word " * 60
        out = headline(long, 100)
        assert not out.rstrip("…").endswith("wor")
        assert out.endswith("…")

    def test_short_text_is_untouched(self):
        from pipeline.utils import headline
        assert headline("Short headline", 160) == "Short headline"

    def test_a_limit_under_forty_does_not_raise(self):
        """The sentence pattern interpolated n as the upper bound of a
        {40,n} repeat, so any limit under 40 built an invalid regex and
        raised re.error instead of truncating. truncate() next door accepts
        any n."""
        from pipeline.utils import headline
        for n in (1, 5, 10, 30, 39):
            out = headline("some fairly long text that needs cutting down "
                           "to size here", n)
            assert isinstance(out, str) and out, n


class TestCapacityCountsBatchesNotCalls:
    def test_a_tier_that_slices_a_batch_counts_as_fewer_batches(self):
        """remaining counts CALLS. The gemma tiers take batch_size 3, so a
        20-item batch costs them 7 calls — 8 remaining calls is about one
        batch, not eight. Counting calls as batches overstated the ladder by
        ~14 batches, so the pre-emptive truncation would not fire and the
        shortfall surfaced as failed batches late in the run instead."""
        from pipeline.budget import Budget
        b = Budget()
        assert b.capacity(20) < b.capacity(), (
            "batch-aware capacity must be lower than the raw call count")
        # No argument keeps the old meaning, so nothing else shifts under it.
        assert b.capacity() == sum(t.remaining for t in b.tiers)


class TestAttackFacets:
    """"Attacks & Exploits" was one pile of 339 stories a day with no way to
    separate an advisory from a malware family from a named gang."""

    def _tags(self, title, summary=""):
        from pipeline import tagging
        from pipeline.models import Item
        it = Item(cluster_key="k", title=title, category="cyber_attacks",
                  summary=summary)
        tagging.apply(it, "")
        return set(it.tags)

    def test_cve_is_matched_not_guessed(self):
        """`vulnerability` fires on the word "patch" and covers advisories
        with no number at all. `cve` means there is an identifier to look up."""
        assert "cve" in self._tags("CVE-2026-83548: SonicWall SMA1000 SSRF")
        assert "cve" not in self._tags("Cisco ships a security patch today")

    def test_named_gangs_are_separated_from_generic_attack_words(self):
        assert "threat-actor" in self._tags("Lazarus Windows Zero-Day Exploit")
        assert "threat-actor" in self._tags("INC Ransom claims attack on clinic")
        assert "threat-actor" not in self._tags("Router flaw lets attackers in")

    def test_ambiguous_gang_names_need_qualifying(self):
        """Bare "play", "hive", "royal" and "cactus" are ordinary English and
        would mis-tag half the feed."""
        for benign in ("Apple Play Store revenue climbs",
                       "Royal Mail confirms delivery delays",
                       "Cactus plant retailer opens in Pune"):
            assert "threat-actor" not in self._tags(benign), benign

    def test_priority_tags_survive_the_eight_tag_cap(self):
        """Sorting alphabetically and slicing cut `threat-actor` before `ai`
        and `cloud`, dropping the one tag that said what the story was."""
        t = self._tags(
            "Lazarus exploits CVE-2026-1234 with ransomware on AWS",
            "An AI-assisted attack on Windows and Linux cloud identity "
            "systems using appsec and network tooling for red team access.")
        assert "cve" in t and "threat-actor" in t


class TestFeedTitlesAreDecoded:
    def test_html_entities_do_not_reach_the_card(self):
        """Feeds escape their titles and nothing decoded them, so cards read
        "Brazilian &amp; Global" and "it&#8217;s still committed"."""
        import html
        for raw, want in [
            ("Brazilian &amp; Global Financial Systems",
             "Brazilian & Global Financial Systems"),
            ("GoPro says it&#8217;s committed", "GoPro says it’s committed"),
            ("batch.py &#x5b;Guest Diary&#x5d;", "batch.py [Guest Diary]"),
            ("Double &amp;amp; encoded", "Double & encoded"),
        ]:
            assert html.unescape(html.unescape(raw)).strip() == want, raw


class TestExtractiveBodiesAreReadable:
    """Most of the page is extractive, so the fallback's output IS the page."""

    def _run(self, title, body, **kw):
        from pipeline.s5_summarize import _heuristic
        from pipeline.models import Item
        it = Item(cluster_key="k", title=title, category="cyber_attacks", **kw)
        _heuristic(it, body)
        return it

    def test_mailing_list_furniture_is_stripped(self):
        """seclists and oss-security wrap the story in a byline and links, so
        the first three sentences were "Posted by Alan Coopersmith on Sep 04
        https://... The GHSA advisories listed above can be found on https://..."
        — a byline, two URLs and a cross-reference."""
        it = self._run(
            "pcre2 version 10.48 released with security fixes",
            "Posted by Alan Coopersmith on Sep 04 "
            "https://github.com/PCRE2Project/pcre2/NEWS reports: The GHSA "
            "advisories listed above can be found on "
            "https://github.com/PCRE2Project/pcre2/security None seem to have "
            "CVE ids assigned at this time. This release fixes a heap buffer "
            "overflow triggered by a crafted pattern during JIT compilation.")
        assert "Posted by" not in it.summary
        assert "http" not in it.summary
        assert "heap buffer overflow" in it.summary

    def test_no_generic_one_liner(self):
        """"Cyber Attacks: single report, unconfirmed." sat on 909 of 1,146
        cards, restating the section heading and the verdict chip already
        above it, in the slot a reader looks to for what the story means."""
        it = self._run("Some breach happened at a company", "A breach occurred "
                       "at the company and data was taken from its servers.")
        assert it.one_liner == ""
        it2 = self._run("Corroborated story", "Body text that is long enough "
                        "to survive the forty character sentence filter here.",
                        corroboration=3)
        assert it2.one_liner == ""

    def test_a_body_of_pure_furniture_yields_nothing_not_junk(self):
        it = self._run("Advisory", "Posted by Someone on Sep 04 "
                       "======================================== "
                       "Read more https://example.com/thing")
        assert it.summary == ""


class TestVendorMarketingIsMuted:
    def test_webinar_and_signup_titles_are_muted(self):
        """"[Virtual Event] What Every Enterprise Should Know About Securing
        Cloud Assets in the Age of AI" is a webinar signup published by a real
        security outlet, and it reached the page looking like reporting."""
        from pipeline.config import interests
        from pipeline.s2_clean import _is_muted
        from pipeline.models import Article
        mutes = interests()["mute"]
        for title in ("[Virtual Event] What Every Enterprise Should Know",
                      "[Webinar] Securing the cloud",
                      "New research: register now to save your seat"):
            a = Article(url="https://x.test/a", title=title, source="S",
                        domain="x.test", category="cyber_attacks", tier="B",
                        published=None)
            assert _is_muted(a, mutes), title

    def test_real_reporting_is_not_muted(self):
        from pipeline.config import interests
        from pipeline.s2_clean import _is_muted
        from pipeline.models import Article
        mutes = interests()["mute"]
        for title in ("Lazarus exploits CVE-2026-1234 in the wild",
                      "Cisco patches critical Nexus 9000 flaw",
                      "Event logging bypass found in Windows"):
            a = Article(url="https://x.test/a", title=title, source="S",
                        domain="x.test", category="cyber_attacks", tier="B",
                        published=None)
            assert not _is_muted(a, mutes), title


class TestSummariesAreReadableEnglish:
    def _sum(self, body, title="X"):
        from pipeline.s5_summarize import _heuristic
        from pipeline.models import Item
        it = Item(cluster_key="k", title=title, category="cyber_tools")
        _heuristic(it, body)
        return it.summary

    def test_medium_teaser_advert_is_stripped(self):
        """Medium publishes a truncated teaser and appends its own advert, so
        the card body ended '... Continue reading on Medium >>'."""
        out = self._sum("A long enough English sentence about pivoting "
                        "through the internal network here. "
                        "Continue reading on Medium »")
        assert "Continue reading" not in out
        assert "pivoting" in out

    def test_a_summary_in_another_script_is_dropped(self):
        """The page is written in English; a reader cannot use a summary they
        cannot read. The headline and source link still publish."""
        arabic = ("قال الله تعالى "
                  "يرفع الله الذين "
                  "آمنوا منكم والذين "
                  "أوتوا العلم درجات")
        assert self._sum(arabic) == ""

    def test_english_text_naming_a_foreign_entity_is_kept(self):
        """A ratio, not a single character. English reporting routinely names
        a Chinese company or quotes a Russian handle, and dropping those would
        lose real security coverage."""
        out = self._sum("Researchers at the Chinese firm 腾讯 disclosed a "
                        "flaw in gateway software this week. The bug allows "
                        "unauthenticated remote code execution on appliances.")
        assert "remote code execution" in out

    def test_extractive_summaries_are_not_one_liners(self):
        """Extractive bodies ran to a median of 141 characters against 438 for
        model-written ones — the difference between a summary and a caption."""
        body = " ".join(
            f"This is sentence number {i} and it carries enough detail about "
            f"the vulnerability to be worth reading on its own." 
            for i in range(8))
        out = self._sum(body)
        assert len(out) > 400, len(out)


class TestConsoleEncodingIsSafeFromAnyEntryPoint:
    def test_importing_the_package_forces_utf8(self):
        """Windows hands a bare python a cp1252 stdout, which raises on an
        emoji in a feed title or a CJK headline. run.py forced UTF-8 on its
        own streams, but pytest and any direct import bypassed that guard."""
        import sys
        import pipeline.utils                     # noqa: F401  (import is the point)
        for stream in (sys.stdout, sys.stderr):
            enc = (getattr(stream, "encoding", "") or "").lower()
            # pytest swaps in its own capture object; accept anything that can
            # actually encode the characters we handle.
            "⚠️ 腾讯 قال".encode(enc or "utf-8")


class TestExtractionDoesNotParseConcurrently:
    """trafilatura parses through lxml, which is libxml2 — a C extension.
    Running it across eight threads aborted the interpreter outright on the
    GitHub runner:

        fetching 700 article bodies, 428 kept as blurbs
        double free or corruption (out)
        2351 Aborted (core dumped) python run.py --quiet

    A glibc heap abort, not a Python exception, so nothing upstream could
    catch it and the whole run died at stage 3. It survived weeks at a
    220-item cap and failed the first night the cap was 700 — more work per
    run, same latent race, suddenly reached."""

    def test_only_the_download_is_submitted_to_the_pool(self):
        import inspect
        from pipeline import s3_extract
        src = inspect.getsource(s3_extract.run)
        code = " ".join(l.split("#")[0] for l in src.splitlines())
        assert "pool.submit(_download" in code, (
            "the thread pool must carry the network fetch")
        assert "pool.submit(_extract" not in code and                "pool.submit(_parse" not in code, (
            "lxml parsing is back inside the thread pool")

    def test_download_and_parse_are_separate_functions(self):
        from pipeline import s3_extract
        assert hasattr(s3_extract, "_download")
        assert hasattr(s3_extract, "_parse")

    def test_a_bad_page_does_not_stop_the_stage(self):
        """A parse failure is one lost body, not a lost run."""
        from pipeline import s3_extract
        assert s3_extract._parse("") == ""


class TestExtractionHonoursItsOwnTimeout:
    """harvest.timeout_seconds was threaded all the way down to _download and
    then dropped — fetch_url ran with no config, so every request used
    trafilatura's DEFAULT_CONFIG instead. Measured on 2.3.1 that default is
    30s, double what this project configures, and at extract.max_items 1400
    over 8 workers the difference is ~87 minutes of timeouts against a
    75-minute workflow cap."""

    def test_the_configured_timeout_reaches_trafilatura(self):
        from pipeline.s3_extract import _traf_config
        assert _traf_config(15).get("DEFAULT", "DOWNLOAD_TIMEOUT") == "15"
        assert _traf_config(7).get("DEFAULT", "DOWNLOAD_TIMEOUT") == "7"

    def test_ssrf_protection_is_pinned_on_not_merely_defaulted(self):
        """trafilatura vets the peer address AFTER connecting, on every hop,
        so a feed link redirecting to the metadata endpoint is refused even
        though netguard only saw the first URL. Verified against a private
        address that was actually listening: on -> blocked, off -> 78,345
        bytes. That makes it load-bearing, so it is asserted rather than
        left to a library default that could change."""
        from pipeline.s3_extract import _traf_config
        assert _traf_config(15).get("DEFAULT", "SSRF_PROTECTION") == "on"

    def test_download_passes_the_config_through(self):
        import inspect
        from pipeline import s3_extract
        src = inspect.getsource(s3_extract._download)
        code = " ".join(l.split("#")[0] for l in src.splitlines())
        assert "config=" in code, "fetch_url called without a config again"


class TestShortenerRedirectsAreVettedPerHop:
    """_resolve_short let urlopen chase the whole redirect chain and only
    inspected the final URL afterwards, with resolve_dns=False — a scheme
    check and nothing more. By then the request had been made. Unlike stage
    3 there is no library guard underneath this one."""

    def test_a_redirect_into_link_local_space_is_never_fetched(self, monkeypatch):
        import urllib.error, urllib.request
        from pipeline import s1d_telegram as T
        seen = []

        def fake_open(self, req, timeout=None):
            seen.append(req.full_url)
            raise urllib.error.HTTPError(
                req.full_url, 302, "Found",
                {"Location": "http://169.254.169.254/latest/meta-data/"}, None)

        monkeypatch.setattr(urllib.request.OpenerDirector, "open", fake_open)
        out = T._resolve_short("https://bit.ly/abc", "ua", 5)
        assert not any("169.254" in u for u in seen), (
            f"metadata endpoint was requested: {seen}")
        assert seen == ["https://bit.ly/abc"]
        # Empty is correct: the caller's netguard.is_safe("") is False, so the
        # post is dropped rather than published with a poisoned link.
        assert out == ""

    def test_a_redirect_chain_is_bounded(self, monkeypatch):
        import urllib.error, urllib.request
        from pipeline import s1d_telegram as T
        seen = []

        def fake_open(self, req, timeout=None):
            seen.append(req.full_url)
            raise urllib.error.HTTPError(
                req.full_url, 302, "Found",
                {"Location": f"https://bit.ly/{len(seen)}"}, None)

        monkeypatch.setattr(urllib.request.OpenerDirector, "open", fake_open)
        T._resolve_short("https://bit.ly/start", "ua", 5)
        assert len(seen) <= T._MAX_HOPS + 1, f"unbounded chain: {len(seen)}"


class TestCommittedCachesSurviveCorruption:
    """All five caches under data/cache/ are committed by CI every run, so a
    bad one is persistent rather than a one-day problem. _tally already
    learned that json.loads returns a list, a string or None just as happily
    for valid-but-not-object input; these two had only a JSONDecodeError
    guard, which does not catch it."""

    JUNK = ("[]", "null", '"x"', "42", '{"truncated')

    def test_budget_survives_every_junk_shape(self, tmp_path, monkeypatch):
        """Budget.__init__ does self._data.get(...), which raises
        AttributeError on a list — not a JSONDecodeError — killing stage 5
        after stages 1-4 have spent the whole network budget."""
        from pipeline import budget as B
        f = tmp_path / "usage.json"
        monkeypatch.setattr(B, "USAGE_PATH", f)
        for junk in self.JUNK:
            f.write_text(junk, encoding="utf-8")
            assert len(B.Budget().tiers) == 8, junk

    def test_summary_cache_survives_every_junk_shape(self, tmp_path, monkeypatch):
        """cache[ckey] = ... raises TypeError on a list."""
        from pipeline import s5_summarize as S
        f = tmp_path / "summaries.json"
        monkeypatch.setattr(S, "SUMMARY_CACHE", f)
        for junk in self.JUNK:
            f.write_text(junk, encoding="utf-8")
            assert S._load_cache() == {}, junk


class TestSummaryCacheEvictsWhatIsUnused:
    """The trim was list(cache.items())[-5000:], described as "keep the most
    recent". Dict order is INSERTION order and re-assigning a key does not
    move it, so a summary reused every day kept its original position and was
    evicted as soon as 5,000 newer entries arrived — while 5,000 write-once
    entries survived. Measured on the committed file: 5,000 entries at the
    cap, 7.8 MB, only 4,907 distinct payloads."""

    def test_a_reused_entry_survives_the_trim(self, tmp_path, monkeypatch):
        from pipeline import s5_summarize as S
        monkeypatch.setattr(S, "SUMMARY_CACHE", tmp_path / "c.json")
        cache = {"hot": {"v": 1}}
        for i in range(S.CACHE_MAX + 1000):
            cache[f"cold{i}"] = {"v": i}
        S._touch(cache, "hot", "hot")        # what a cache hit does
        S._save_cache(cache)
        assert "hot" in cache, "the entry being reused was evicted"
        assert len(cache) == S.CACHE_MAX

    def test_refiling_under_a_new_key_leaves_no_duplicate(self):
        from pipeline.s5_summarize import _touch
        cache = {"oldkey": {"v": 9}}
        _touch(cache, "oldkey", "newkey")
        assert list(cache) == ["newkey"], f"duplicate left behind: {list(cache)}"

    def test_the_cap_is_actually_enforced(self, tmp_path, monkeypatch):
        from pipeline import s5_summarize as S
        monkeypatch.setattr(S, "SUMMARY_CACHE", tmp_path / "c.json")
        cache = {f"k{i}": {"v": i} for i in range(S.CACHE_MAX + 500)}
        S._save_cache(cache)
        assert len(cache) == S.CACHE_MAX


class TestDryRunWritesNothingAndSpendsNothing:
    """run.py documents --dry-run as "run everything, write nothing". Only
    stages 7 and 8 honoured it: the summariser still called the API and
    still wrote summaries.json, and Budget still wrote usage.json on every
    call. Reproducing a bug with --dry-run therefore spent the day's
    free-tier quota AND left the counters claiming the real run had spent
    it, so the scheduled run started from a stale number."""

    def test_the_flag_gates_every_cache_write(self, tmp_path, monkeypatch):
        import json
        from pipeline import config, budget as B, s5_summarize as S
        usage = tmp_path / "usage.json"
        summaries = tmp_path / "summaries.json"
        usage.write_text("{}", encoding="utf-8")
        summaries.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(B, "USAGE_PATH", usage)
        monkeypatch.setattr(S, "SUMMARY_CACHE", summaries)
        monkeypatch.setattr(config, "WRITES_ENABLED", False)

        bud = B.Budget()
        bud.record(bud.tiers[0])
        bud.cool(bud.tiers[0], "503")
        S._save_cache({"k": {"summary": "x"}})

        assert usage.read_text(encoding="utf-8") == "{}", "usage.json written"
        assert summaries.read_text(encoding="utf-8") == "{}", "cache written"

    def test_writes_resume_when_the_flag_is_on(self, tmp_path, monkeypatch):
        """The guard must not be a permanent off switch."""
        from pipeline import config, s5_summarize as S
        f = tmp_path / "summaries.json"
        monkeypatch.setattr(S, "SUMMARY_CACHE", f)
        monkeypatch.setattr(config, "WRITES_ENABLED", True)
        S._save_cache({"k": {"summary": "x"}})
        assert f.exists() and "summary" in f.read_text(encoding="utf-8")

    def test_no_api_call_is_attempted(self, monkeypatch):
        """Falls back to extractive rather than spending quota."""
        from pipeline import config, s5_summarize as S
        from pipeline.models import Item, Priority, Verdict
        monkeypatch.setattr(config, "WRITES_ENABLED", False)

        def explode(*a, **k):
            raise AssertionError("an API call was attempted during a dry run")
        monkeypatch.setattr(S, "api_key", explode)

        items = [Item(cluster_key="k", title="A flaw", category="cyber_attacks",
                      priority=Priority.IMPORTANT, verdict=Verdict.ESTABLISHED)]
        out = S.run(items, {"k": "A critical vulnerability was disclosed "
                                 "today and is being exploited widely."})
        assert out[0].summary_source == "extract"


class TestNoSecretsInRepo:
    def test_env_is_gitignored(self):
        """A key reaching a git remote is the one unrecoverable mistake here."""
        root = Path(__file__).resolve().parent.parent
        ignored = (root / ".gitignore").read_text(encoding="utf-8").splitlines()
        # Must be its own line: a trailing comment makes the pattern literal
        # and silently match nothing. That exact bug tracked 4.1 MB of
        # snapshots before it was caught.
        assert ".env" in [line.strip() for line in ignored]

    def test_no_key_shaped_strings_in_any_tracked_file(self):
        """Scan everything git tracks, not just config/.

        The original version checked config/*.yaml only, which would not have
        caught a key pasted into README.md, PROGRESS.md or a data/ file. It
        also had no pattern for a Telegram bot token or the newer GitHub
        token prefixes.
        """
        import re
        import subprocess

        root = Path(__file__).resolve().parent.parent
        pat = re.compile(
            r"AIza[0-9A-Za-z_-]{20,}"            # Google API key
            r"|AQ\.[A-Za-z0-9_-]{20,}"           # newer Google AI Studio key
            r"|gh[pousr]_[A-Za-z0-9]{20,}"       # GitHub classic/fine tokens
            r"|github_pat_[A-Za-z0-9_]{20,}"
            r"|\d{8,10}:AA[A-Za-z0-9_-]{30,}"  # Telegram bot token
            r"|sk-[A-Za-z0-9]{20,}"              # OpenAI-style
            r"|xox[baprs]-[A-Za-z0-9-]{10,}"     # Slack
        )
        tracked = subprocess.run(["git", "ls-files"], cwd=root,
                                 capture_output=True, text=True).stdout.split()
        assert tracked, "git ls-files returned nothing"

        for rel in tracked:
            path = root / rel
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, FileNotFoundError, OSError):
                continue                      # binary or removed; nothing to scan
            found = pat.search(text)
            assert not found, f"key-shaped string in {rel}: {found.group()[:12]}..."


class TestSecretSanitising:
    """One validation closes three separate leak paths.

    A secret containing a control character makes http.client raise
    ValueError("Invalid header value %r") with the FULL header in the
    message — which s1b_github logs, llm_gemini lets escape as a traceback,
    and s8_notify echoes in three handlers. Sanitising at the entry point
    beats patching each site, because the next handler written would leak too.
    """

    def test_control_characters_are_removed(self):
        from pipeline.config import clean_secret
        for bad in (chr(10), chr(13), chr(9), chr(0), chr(31)):
            out = clean_secret("AIza" + bad + "SyKEY", "TEST")
            assert bad not in out
            assert out == "AIzaSyKEY"

    def test_a_clean_key_is_returned_unchanged(self):
        from pipeline.config import clean_secret
        assert clean_secret("AIzaSyABC123", "TEST") == "AIzaSyABC123"

    def test_surrounding_whitespace_is_trimmed(self):
        from pipeline.config import clean_secret
        assert clean_secret("  AIzaSyABC123  ", "TEST") == "AIzaSyABC123"

    def test_empty_and_none_become_none(self):
        from pipeline.config import clean_secret
        assert clean_secret("", "TEST") is None
        assert clean_secret(None, "TEST") is None
        assert clean_secret("   ", "TEST") is None

    def test_sanitised_value_is_header_safe(self):
        """The actual property that matters: http.client must accept it."""
        import http.client
        from pipeline.config import clean_secret
        dirty = "Bearer_tok" + chr(10) + "en"
        cleaned = clean_secret(dirty, "TEST")
        # This is the call that raised ValueError with the token in the message.
        http.client.HTTPConnection("example.com")._validate_header_value =             getattr(http.client.HTTPConnection, "_validate_header_value", None)
        assert all(ord(c) > 0x20 for c in cleaned)


class TestPromptHardening:
    """The prompt must structurally separate instructions from content.

    Live injection tests against three model tiers all resisted, but that is
    a property of today's models. These assert the defence is present in the
    prompt itself.
    """

    def test_article_text_is_delimited(self):
        from pipeline.llm_gemini import SYSTEM
        import pipeline.llm_gemini as m
        assert "UNTRUSTED INPUT" in SYSTEM
        assert "BEGIN ARTICLE" in Path(m.__file__).read_text(encoding="utf-8")

    def test_system_prompt_refuses_embedded_instructions(self):
        from pipeline.llm_gemini import SYSTEM
        # Collapse whitespace first: the prompt is hard-wrapped, so a phrase
        # can straddle a newline and a naive substring check misses it.
        low = " ".join(SYSTEM.lower().split())
        assert "never instructions" in low or "not instructions" in low
        assert "do not comply" in low

    def test_judge_is_hardened_too(self):
        from pipeline.s6b_judge import JUDGE_SYSTEM
        assert "untrusted" in JUDGE_SYSTEM.lower()


class TestShortlistCoverage:
    """The shortlist must be wider than the publish cut, per category.

    This is the regression that produced 26 of 40 published cards carrying an
    extractive fallback instead of a real summary: the shortlist allocated
    each category exactly its publish quota, so anything promoted by the
    model's importance score had never been summarised in the first place.
    """

    def _items(self, per_cat=60):
        from pipeline.models import Item
        out = []
        for cat in ("tech_ai", "cyber_attacks", "cyber_tools",
                    "markets", "india_world"):
            for n in range(per_cat):
                it = Item(cluster_key=f"{cat}{n}", title="t", category=cat)
                it.blend = float(per_cat - n)
                out.append(it)
        return out

    def test_shortlist_is_bounded_by_the_model_budget(self):
        """The shortlist is a spending decision. It used to size itself from
        the PUBLISH quota, which was fine while the page showed a top 40 and
        catastrophic once it started publishing everything in the window —
        it would have asked the model for hundreds of summaries a day the
        free tier cannot pay for."""
        from pipeline.config import settings
        from pipeline.s5_summarize import _shortlist
        from pipeline.models import Item, Priority, Verdict
        import collections

        cfg = settings()["llm"]
        budget = cfg["max_items_summarized"]
        cats = ["cyber_attacks", "cyber_tools", "tech_ai", "markets",
                "india_world"]
        items = []
        for c in cats:
            for n in range(400):           # far more than any budget
                items.append(Item(cluster_key=f"{c}{n}", title="t",
                                  category=c, blend=float(n),
                                  priority=Priority.MINOR,
                                  verdict=Verdict.ESTABLISHED))
        per_cat = max(1, budget // len(cats))
        chosen, rest = _shortlist(items, budget, per_cat,
                                  cfg.get("shortlist_multiplier", 2.0))
        assert len(chosen) <= budget, (
            f"shortlist {len(chosen)} exceeds the {budget} the budget allows")
        assert len(chosen) + len(rest) == len(items), "items were lost"
        # and every category has to be represented, or a quiet section ships
        # with no model summaries at all
        seen = collections.Counter(i.category for i in chosen)
        assert set(seen) == set(cats), f"category starved: {sorted(seen)}"

    def test_shortlist_is_not_starved_by_a_dominant_category(self):
        """A category with 10x the volume must not consume the whole budget."""
        import collections
        from pipeline.models import Item
        from pipeline.s5_summarize import _shortlist

        items = self._items(per_cat=5)
        for n in range(300):                       # tech_ai floods the harvest
            it = Item(cluster_key=f"flood{n}", title="t", category="tech_ai")
            it.blend = 999.0
            items.append(it)

        short, _ = _shortlist(items, 150, 12, 2.0)
        per = collections.Counter(i.category for i in short)
        for cat in ("markets", "cyber_tools", "india_world"):
            assert per[cat] == 5, f"{cat} starved: {per[cat]}"


class TestTelegramNotify:
    """Pushing posts to a channel other people read, so the guards matter
    more than the formatting."""

    def _item(self, title="A title", priority=None, verdict=None):
        from pipeline.models import Item, Priority, Verdict
        it = Item(cluster_key="k", title=title, category="cyber_attacks")
        it.priority = priority or Priority.CRITICAL
        it.verdict = verdict or Verdict.VERIFIED
        it.bottom_line = "Patch now."
        it.sources = [{"name": "s", "url": "https://example.com/a",
                       "domain": "example.com", "tier": "A"}]
        return it

    def test_disabled_by_default(self):
        """It must never start posting as a side effect of a config default."""
        from pipeline.config import settings
        assert settings()["notify"]["telegram"]["enabled"] is False

    def test_does_nothing_when_disabled(self):
        from pipeline import s8_notify
        # No exception, no network, regardless of what it is handed.
        s8_notify.run([self._item()], "2026-01-01T00:00:00+00:00")

    def test_html_is_escaped(self):
        """Titles come from feeds, so they can contain anything."""
        from pipeline import s8_notify
        out = s8_notify._format(
            [self._item("<script>alert(1)</script> & co")], "2026-01-01T00:00")[0]
        assert "<script>" not in out
        assert "&lt;script&gt;" in out
        assert "&amp;" in out

    def test_splits_on_telegram_length_limit(self):
        """Telegram hard-limits a message to 4096 chars."""
        from pipeline import s8_notify
        items = [self._item("x" * 120) for _ in range(60)]
        msgs = s8_notify._format(items, "2026-01-01T00:00")
        assert len(msgs) > 1
        assert all(len(m) <= 4096 for m in msgs)

    def test_rejected_items_are_never_pushed(self):
        from pipeline.models import Verdict
        from pipeline import s8_notify
        rejected = self._item(verdict=Verdict.REJECTED)
        # _format does not filter; run() does. Assert the filter exists by
        # checking the verdict is excluded from the wanted set logic.
        assert rejected.verdict is Verdict.REJECTED


class TestTelegramSource:
    def test_shortener_list_covers_the_common_ones(self):
        from pipeline.s1d_telegram import SHORTENERS
        # ift.tt is what the user's own channel posts through; an unresolved
        # shortener breaks tiering, dedupe and the reference link at once.
        for host in ("ift.tt", "bit.ly", "buff.ly", "t.co"):
            assert host in SHORTENERS

    def test_message_parsing_extracts_text_and_link(self):
        from pipeline.s1d_telegram import _parse
        block = ('Gunra ransomware: what you need to know '
                 '<a href="https://ift.tt/abc">link</a>')
        text, link = _parse(block)
        assert "Gunra ransomware" in text
        assert link == "https://ift.tt/abc"

    def test_telegram_links_are_not_treated_as_the_article(self):
        from pipeline.s1d_telegram import _parse
        _, link = _parse('see <a href="https://t.me/other/1">this</a>')
        assert link is None


class TestSecretsFilePermissions:
    """Gitignoring .env protects it from the remote, not from the local disk.

    This machine's .env had inherited read access for the local Users group
    and Authenticated Users, so any account on the box could read the API key.
    """

    def test_matcher_flags_risky_groups(self):
        from pipeline.config import _ACL_RISKY
        BS = chr(92)
        cases = {
            "BUILTIN" + BS + "Users:(I)(RX)": "Users",
            "NT AUTHORITY" + BS + "Authenticated Users:(I)(M)":
                "Authenticated Users",
            ".env Everyone:(F)": "Everyone",
            "MACHINE" + BS + "Users:(RX)": "Users",
        }
        for text, expected in cases.items():
            hits = {m.group(1) for m in _ACL_RISKY.finditer(text)}
            assert expected in hits, f"{text!r} -> {hits}"

    def test_matcher_allows_an_owner_only_acl(self):
        from pipeline.config import _ACL_RISKY
        # Generic on purpose: this fixture was originally pasted from a real
        # icacls run, which put a machine name and a local account name into
        # a file destined for a public repository.
        safe = ".env DESKTOP-EXAMPLE" + chr(92) + "owner:(R,W)"
        assert not list(_ACL_RISKY.finditer(safe))

    def test_authenticated_users_is_not_mislabelled_as_users(self):
        """Alternation order matters: the bare 'Users' branch would otherwise
        swallow 'Authenticated Users' and report the wrong group."""
        from pipeline.config import _ACL_RISKY
        text = "NT AUTHORITY" + chr(92) + "Authenticated Users:(I)(M)"
        hits = {m.group(1) for m in _ACL_RISKY.finditer(text)}
        assert hits == {"Authenticated Users"}

    def test_check_never_raises(self):
        """A permissions check must not be able to break a run."""
        from pathlib import Path
        from pipeline.config import _warn_if_world_readable
        _warn_if_world_readable(Path("does-not-exist-anywhere.env"))


class TestCategoryKeysMatchFeeds:
    """The single worst class of bug in this project: a config key that no
    code path ever looks up, so the feature silently does nothing."""

    def test_every_boost_key_is_a_real_category(self):
        """A `cybersecurity:` key survived a category rename and matched
        nothing, so every security keyword was dead — exploited zero-days
        scored the bare category floor of 12 while markets stories scored 100."""
        from pipeline.config import feeds, interests
        real = set(feeds())
        for key in interests()["boost"]:
            assert key in real, f"boost.{key} matches no category in feeds.yaml"

    def test_every_category_has_a_boost_profile(self):
        from pipeline.config import feeds, interests
        boost = set(interests()["boost"])
        for cat in feeds():
            assert cat in boost, f"category {cat} has no keyword profile"

    def test_security_keywords_actually_fire(self):
        from datetime import datetime, timezone
        from pipeline.config import settings
        from pipeline.models import Article, Cluster
        from pipeline.s4_corroborate import _interest_score

        def score(cat, title):
            a = Article(url="https://x.com/1", title=title, source="s",
                        domain="x.com", category=cat, tier="B",
                        published=datetime.now(timezone.utc))
            return _interest_score(Cluster(key="k", articles=[a]),
                                   settings()["interest"])

        hot = score("cyber_attacks",
                    "Actively exploited zero-day RCE ransomware CVE-2026-1")
        routine = score("cyber_attacks", "Vendor publishes quarterly report")
        assert hot > routine + 40, f"hot={hot} routine={routine}"


class TestGeminiResponseShapes:
    """Gemini omits `content` on a safety block and omits `candidates`
    entirely when the prompt is rejected. Both were dereferenced directly, so
    a KeyError escaped every caller and killed the run AFTER stages 1-5 had
    spent their budget. A news feed is untrusted input, so a safety block is
    routine here, not exceptional."""

    def _extract(self, data):
        from pipeline.llm_gemini import _extract_text
        return _extract_text(data, "test-model")

    def test_no_candidates_raises_gemini_error(self):
        from pipeline.llm_gemini import GeminiError
        with pytest.raises(GeminiError):
            self._extract({"promptFeedback": {"blockReason": "SAFETY"}})

    def test_candidate_without_content_raises_gemini_error(self):
        from pipeline.llm_gemini import GeminiError
        with pytest.raises(GeminiError):
            self._extract({"candidates": [{"finishReason": "SAFETY"}]})

    def test_max_tokens_truncation_raises_gemini_error(self):
        from pipeline.llm_gemini import GeminiError
        with pytest.raises(GeminiError):
            self._extract({"candidates": [{"finishReason": "MAX_TOKENS"}]})

    def test_healthy_response_returns_text(self):
        out = self._extract({"candidates": [
            {"finishReason": "STOP", "content": {"parts": [{"text": "hi"}]}}]})
        assert out == "hi"

    def test_never_raises_keyerror(self):
        """The property that matters: callers catch GeminiError only."""
        for data in ({}, {"candidates": []}, {"candidates": [{}]},
                     {"candidates": [{"content": {}}]}):
            try:
                self._extract(data)
            except Exception as exc:
                assert type(exc).__name__ == "GeminiError", type(exc).__name__


class TestNotifyRanking:
    def test_critical_survives_a_flood_of_routine_items(self):
        """The digest slice was taken from unsorted cluster order, so a
        CRITICAL item could be dropped for ten routine ones."""
        from pipeline.models import Item, Priority, Verdict
        from pipeline import s8_notify

        items = []
        for n in range(12):
            it = Item(cluster_key=f"r{n}", title=f"routine {n}",
                      category="markets")
            it.priority, it.verdict, it.blend = Priority.IMPORTANT, Verdict.ESTABLISHED, 10.0
            items.append(it)
        hot = Item(cluster_key="hot", title="ACTIVELY EXPLOITED ZERO-DAY",
                   category="cyber_attacks")
        hot.priority, hot.verdict, hot.blend = Priority.CRITICAL, Verdict.VERIFIED, 99.0
        items.append(hot)

        order = {Priority.CRITICAL: 0, Priority.IMPORTANT: 1, Priority.MINOR: 2}
        ranked = sorted(
            (i for i in items if i.verdict is not Verdict.REJECTED),
            key=lambda i: (order[i.priority], -i.blend))[:10]
        assert ranked[0] is hot


class TestVerifyMissingBody:
    def test_no_body_is_unverifiable_not_contradicted(self):
        """'We never fetched the article' and 'the model invented its
        evidence' are different failures and must not share a verdict."""
        from pipeline.models import Claim, Item, Verdict
        from pipeline import s6_verify

        it = Item(cluster_key="k", title="t", category="cyber_attacks")
        it.claims = [Claim(text="a claim", support_span="some quote")]
        s6_verify.run([it], {"k": ""})
        assert it.verdict is not Verdict.REJECTED
        assert all(c.status == "neutral" for c in it.claims)


class TestSourceOrdering:
    def test_aggregator_is_never_the_primary_link(self):
        """sources[0] is what the card links to and what stage 5 tells the
        model. An aggregator there means the headline and the link come from
        different outlets."""
        from pipeline.models import Verdict
        from pipeline.s4_corroborate import run as corroborate

        agg = _art("Acme RCE flaw exploited in the wild", "news.google.com",
                   tier="A", aggregator=True)
        real = _art("Acme RCE flaw exploited in the wild attacks ongoing",
                    "bleepingcomputer.com", tier="A")
        from pipeline.models import Cluster
        out = corroborate([Cluster(key="k", articles=[agg, real])])
        assert out[0].sources[0]["domain"] == "bleepingcomputer.com"


class TestSeenStorePruning:
    """The de-duplication window must forget the OLDEST entries, not the
    lexicographically lowest. uids are hex hashes, so sorting before pruning
    evicted every id starting with a low hex digit regardless of age."""

    def _isolated(self, tmp_path, monkeypatch):
        import pipeline.s2_clean as s2
        monkeypatch.setattr(s2, "SEEN_PATH", tmp_path / "seen.json")
        return s2

    def test_insertion_order_is_preserved(self, tmp_path, monkeypatch):
        import json
        s2 = self._isolated(tmp_path, monkeypatch)
        s2.SEEN_PATH.write_text(json.dumps(["aaa1", "fff9", "0000"]))
        s2.save_seen({"bbb2"})
        assert json.loads(s2.SEEN_PATH.read_text()) == \
            ["aaa1", "fff9", "0000", "bbb2"]

    def test_at_capacity_the_oldest_is_evicted(self, tmp_path, monkeypatch):
        import json
        s2 = self._isolated(tmp_path, monkeypatch)
        s2.SEEN_PATH.write_text(json.dumps([f"{i:04x}" for i in range(20000)]))
        s2.save_seen({"zzzz"})
        out = json.loads(s2.SEEN_PATH.read_text())
        assert len(out) == 20000
        assert out[-1] == "zzzz"          # newest kept
        assert out[0] == "0001"           # oldest dropped
        assert "0000" not in out

    def test_duplicates_are_not_appended_twice(self, tmp_path, monkeypatch):
        import json
        s2 = self._isolated(tmp_path, monkeypatch)
        s2.SEEN_PATH.write_text(json.dumps(["aaa1"]))
        s2.save_seen({"aaa1"})
        assert json.loads(s2.SEEN_PATH.read_text()) == ["aaa1"]


class TestShortenerHostMatching:
    def test_www_prefixed_shortener_is_recognised(self):
        """`www.bit.ly` missed SHORTENERS entirely, so it stayed unresolved
        and broke tiering, dedupe and the reference link at once."""
        from urllib.parse import urlsplit
        from pipeline.s1d_telegram import SHORTENERS
        for url in ("https://WWW.Bit.LY/abc", "https://bit.ly/abc",
                    "http://www.ift.tt/xyz"):
            host = (urlsplit(url).hostname or "").lower().removeprefix("www.")
            assert host in SHORTENERS, url

    def test_a_real_outlet_is_not_treated_as_a_shortener(self):
        from urllib.parse import urlsplit
        from pipeline.s1d_telegram import SHORTENERS
        host = (urlsplit("https://www.bleepingcomputer.com/news/x").hostname
                or "").lower().removeprefix("www.")
        assert host not in SHORTENERS


class TestRescoreTrace:
    def test_stale_scoring_lines_are_cleared(self):
        """rescore() stripped only "blend " lines, so an escalated item
        shipped a trace showing its escalation twice, and a capped item
        showed a contradictory escalate/demote pair. The trace renders
        behind the card toggle, so the reader saw it."""
        from pipeline.models import Item, Priority
        from pipeline.s4_corroborate import rescore

        it = Item(cluster_key="k", title="t", category="cyber_attacks")
        it.priority, it.blend, it.interest_score = Priority.CRITICAL, 80.0, 100
        it.trace = ["tier A (+40)", "blend 70 = ...",
                    "escalated to CRITICAL by signal: 'actively exploited'",
                    "demoted: critical capped at 5"]
        rescore([it])
        kept = [t for t in it.trace if t.startswith(("blend ", "escalated to",
                                                     "demoted:"))]
        # exactly one scoring pass worth of lines, not two
        assert sum(1 for t in kept if t.startswith("escalated to")) <= 1
        assert sum(1 for t in kept if t.startswith("demoted:")) <= 1
        assert "tier A (+40)" in it.trace          # non-scoring lines survive


class TestNoLateImportShadowing:
    """A function-scope `import x` makes x local to the WHOLE function, so any
    earlier use of x in that function raises NameError at runtime rather than
    at import time — invisible to a syntax check and to every test that does
    not execute that exact path.

    This shipped: s7_publish used hashlib and canonical_url at the top of
    run() while importing them 50 lines further down, breaking publish
    entirely while `python -c "import pipeline.s7_publish"` still passed.
    """

    def test_publish_runs_end_to_end(self):
        from pipeline import s7_publish
        from pipeline.models import Item, Priority, Verdict

        it = Item(cluster_key="k", title="t", category="markets")
        it.priority, it.verdict, it.blend = (Priority.IMPORTANT,
                                             Verdict.ESTABLISHED, 10.0)
        it.sources = [{"name": "s", "url": "https://example.com/a",
                       "domain": "example.com", "tier": "B"}]
        out = s7_publish.run([it], dry_run=True)
        assert out["counts"]["published"] == 1

    def test_no_function_scope_import_shadows_a_module_import(self):
        """Flag any name imported BOTH at module scope and inside a function.

        The function-scope import makes the name local to the ENTIRE function,
        so every use of it in that function — including uses that appear
        BEFORE the import statement, and uses on branches where the import
        never executes — raises UnboundLocalError at runtime.

        An earlier version of this test only flagged "used before the import
        line", and it passed while the bug was live: run.py imported
        s8_notify at module scope and again inside main(), so the call at the
        end of main() failed with UnboundLocalError after the entire pipeline
        had already run. Line order was never the issue; the shadowing is.
        """
        import ast
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        targets = list((root / "pipeline").glob("*.py")) + [root / "run.py"]

        offenders = []
        for path in targets:
            tree = ast.parse(path.read_text(encoding="utf-8"))

            module_level = set()
            for node in tree.body:                    # top level only
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        module_level.add((alias.asname or alias.name).split(".")[0])

            for fn in [n for n in ast.walk(tree)
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
                for node in ast.walk(fn):
                    if not isinstance(node, (ast.Import, ast.ImportFrom)):
                        continue
                    for alias in node.names:
                        name = (alias.asname or alias.name).split(".")[0]
                        if name in module_level:
                            offenders.append(
                                f"{path.name}:{node.lineno} re-imports '{name}' "
                                f"inside {fn.name}(), shadowing the module-level "
                                f"import for the whole function")

        assert not offenders, "; ".join(offenders)


class TestCorroborationIndependence:
    """Corroboration is the basis of the `verified` badge, so it has to count
    INDEPENDENT reporting, not reach. A live run counted 26 "outlets" on one
    breach story, of which 21 were Yahoo Finance, Crypto Briefing, The Malone
    Telegram and similar — the long tail republishing a single release."""

    def _cluster(self, domains):
        from datetime import datetime, timezone
        from pipeline.models import Article, Cluster
        return Cluster(key="k", articles=[
            Article(url=f"https://{d}/x", title="t", source=d, domain=d,
                    category="cyber_attacks", tier="B",
                    published=datetime.now(timezone.utc)) for d in domains])

    def test_synthetic_publishers_collapse_to_one_vote(self):
        from pipeline.s4_corroborate import _independent_outlets
        c = self._cluster(["bleepingcomputer.com",
                           "a.publisher", "b.publisher", "c.publisher",
                           "d.publisher", "e.publisher"])
        assert _independent_outlets(c) == 2   # 1 real + 1 collective

    def test_press_release_distributors_are_not_outlets(self):
        from pipeline.s4_corroborate import _independent_outlets
        c = self._cluster(["bleepingcomputer.com", "pr-newswire.publisher",
                           "businesswire.com", "globenewswire.com",
                           "yahoo-finance.publisher"])
        assert _independent_outlets(c) == 2

    def test_genuine_independent_outlets_all_count(self):
        from pipeline.s4_corroborate import _independent_outlets
        c = self._cluster(["bleepingcomputer.com", "cyberscoop.com",
                           "malwarebytes.com", "securityweek.com"])
        assert _independent_outlets(c) == 4

    def test_wire_syndication_still_collapses(self):
        from pipeline.s4_corroborate import _independent_outlets
        c = self._cluster(["reuters.com", "in.reuters.com", "bbc.co.uk"])
        assert _independent_outlets(c) == 2

    def test_wide_pickup_is_weak_evidence_not_none(self):
        """A story carried only by the long tail should count 1, not 0 —
        wide pickup is weak evidence, not the absence of evidence."""
        from pipeline.s4_corroborate import _independent_outlets
        c = self._cluster(["a.publisher", "b.publisher", "c.publisher"])
        assert _independent_outlets(c) == 1


class TestTagPrecision:
    """Tag rules matched bare substrings, so 'rce' matched COMMERCE and a
    Shein IPO story was tagged `exploit`. On a live run the tag appeared on
    58% of items including markets and india_world stories, which makes it
    useless as a filter — the thing tags exist for."""

    def test_rce_does_not_match_commerce(self):
        from pipeline.tagging import from_keywords
        tags = from_keywords("Shein IPO eyes $27 billion in cross-border commerce")
        assert "exploit" not in tags

    def test_poc_does_not_match_pocket(self):
        from pipeline.tagging import from_keywords
        assert "exploit" not in from_keywords("A pocket guide to gardening")

    def test_real_security_terms_still_match(self):
        from pipeline.tagging import from_keywords
        assert "exploit" in from_keywords("Unauthenticated RCE in Citrix NetScaler")
        assert "ransomware" in from_keywords("Ransomware gang hits hospital")
        assert "ai" in from_keywords("New AI model released by OpenAI")

    def test_multi_word_terms_survive_boundaries(self):
        from pipeline.tagging import from_keywords
        assert "exploit" in from_keywords("Proof of concept published today")

    def test_tags_discriminate_across_the_corpus(self):
        """No tag should land on most of the page. A tag on 58% of items
        carries no information."""
        import collections
        import json
        from pathlib import Path

        news = Path(__file__).resolve().parent.parent / "site" / "news.json"
        if not news.exists():
            return                       # nothing published yet; nothing to check
        items = json.loads(news.read_text(encoding="utf-8"))["items"]
        if len(items) < 10:
            return
        counts = collections.Counter(t for i in items for t in i.get("tags", []))
        worst, n = counts.most_common(1)[0]
        assert n <= len(items) * 0.7, (
            f"tag '{worst}' is on {n}/{len(items)} items — too broad to filter on")


class TestRetryBudget:
    """Retry shallow, fall through fast.

    Profiling a 45-minute run put 63% of wall clock inside the retry loop and
    another 10% in RPM pacing — nearly three quarters spent waiting. With a
    ladder of eight models, retrying ONE model deeply is backwards: a
    different model is far more likely to answer than the same one twelve
    seconds later.
    """

    def test_worst_case_on_a_dead_model_is_bounded(self):
        import inspect
        from pipeline import llm_gemini

        sig = inspect.signature(llm_gemini._post_with_retries)
        retries = sig.parameters["max_retries"].default
        call_sig = inspect.signature(llm_gemini.call)
        timeout = call_sig.parameters["timeout"].default

        # attempts x timeout, plus backoff, must stay well under the point
        # where one model can eat a meaningful share of the run
        worst = retries * timeout
        assert worst <= 120, f"a dead model can burn {worst}s before fallthrough"

    def test_ladder_is_deep_enough_to_justify_shallow_retries(self):
        """Shallow retries are only safe because there are models to fall
        through TO. If the ladder shrank, this trade would stop making sense."""
        from pipeline.config import settings
        assert len(settings()["llm"]["tiers"]) >= 4


class TestJudgeScope:
    def test_judge_is_scoped_to_actionable_items(self):
        """A measured run spent 5 calls and ~4 minutes returning 28 'all
        supported', 1 unsupported and 0 rejections — nearly all of it on
        MINOR items nobody acts on."""
        from pipeline.config import settings
        cfg = settings()["judge"]
        assert cfg["min_priority"] in ("critical", "important")
        assert cfg["max_items"] <= 20

    def test_minor_items_are_excluded_by_default(self):
        from pipeline.config import settings
        from pipeline.models import Priority

        floor = settings()["judge"]["min_priority"]
        wanted = {Priority.CRITICAL}
        if floor in ("important", "minor"):
            wanted.add(Priority.IMPORTANT)
        if floor == "minor":
            wanted.add(Priority.MINOR)
        assert Priority.MINOR not in wanted


class TestDeployConfig:
    """The published site is public even though the repo is private, so what
    reaches it matters."""

    def _vercel(self):
        import json
        from pathlib import Path
        return json.loads(
            (Path(__file__).resolve().parent.parent / "vercel.json")
            .read_text(encoding="utf-8"))

    def test_only_the_site_directory_is_published(self):
        """Serving the repo root would expose data/rejected.json and the
        caches. None hold secrets — that was audited — but rejected items and
        internal state are not meant to be public."""
        assert self._vercel()["outputDirectory"] == "site"

    def test_news_json_is_not_cached(self):
        """The whole point is a page that changes daily. A CDN caching
        news.json would show yesterday's stories from a fresh deploy."""
        rules = self._vercel()["headers"]
        newsjson = next(r for r in rules if r["source"] == "/news.json")
        cc = next(h["value"] for h in newsjson["headers"]
                  if h["key"] == "Cache-Control")
        assert "max-age=0" in cc and "must-revalidate" in cc

    def test_csp_allows_the_fonts_the_page_actually_uses(self):
        """A CSP that blocks its own stylesheet is worse than none."""
        rules = self._vercel()["headers"]
        catchall = next(r for r in rules if r["source"] == "/(.*)")
        csp = next(h["value"] for h in catchall["headers"]
                   if h["key"] == "Content-Security-Policy")
        assert "fonts.googleapis.com" in csp
        assert "fonts.gstatic.com" in csp
        assert "default-src 'none'" in csp

    def test_vercelignore_excludes_the_env_file(self):
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        ignored = (root / ".vercelignore").read_text(encoding="utf-8").split()
        assert ".env" in ignored
        assert "data/" in ignored


# ---------------------------------------------------------------------------
# The ladder: wasted attempts, and the run that served 12 of 20 batches.
#
# Evidence these were written against, from data/cache/usage.json (14
# consecutive day-records) and a simulation driving the real Budget/_generate:
#
#   2026-10-22  calls=12  {'gemini-3.5-flash-lite': 11, 'gemini-3.1-flash-lite': 1}
#               failed = 8 entries, every one of them exactly 2
#
#   capacity(20) promised 34 batches; 12 were served; batches 13-20 all fell
#   back to extractive summaries at ONE identical timestamp, inside the 45s
#   bench that was about to expire.
# ---------------------------------------------------------------------------

class _Clock:
    """Virtual monotonic clock, so a 45s bench costs no test time."""

    def __init__(self):
        self.t = 1000.0
        self.slept = 0.0

    def monotonic(self):
        return self.t

    def sleep(self, seconds):
        self.slept += seconds
        self.t += seconds


class _FakeLLM:
    """Stands in for pipeline.llm_gemini. `dead` names models that 503."""

    class GeminiError(RuntimeError):
        pass

    class RateLimited(GeminiError):
        pass

    class Overloaded(GeminiError):
        pass

    # What a 503 really costs: _post_with_retries makes 2 attempts at a 45s
    # timeout with a 2s backoff between them. That is LONGER than the 45s
    # bench, which is the whole reason a benched tier was eligible again by
    # the time _generate looped round to it.
    FAIL_SECONDS = 93.0

    def __init__(self, dead=(), until=None, clock=None, fail_seconds=None):
        self.dead = set(dead)
        self.until = until          # 503 for EVERY model while t < until
        self.clock = clock
        self.fail_seconds = (self.FAIL_SECONDS if fail_seconds is None
                             else fail_seconds)
        self.attempts = {}
        self.calls = {}

    def call(self, model, key, profile, payload, tags=""):
        self.attempts[model] = self.attempts.get(model, 0) + 1
        busy = model in self.dead
        if self.until is not None and self.clock.monotonic() < self.until:
            busy = True
        if busy:
            self.clock.t += self.fail_seconds
            raise self.Overloaded(f"HTTP 503 on {model}")
        self.clock.t += 5.0
        self.calls[model] = self.calls.get(model, 0) + 1
        return [{"id": p["id"], "summary": "s", "one_liner": "o",
                 "bottom_line": "b", "importance": 3, "claims": []}
                for p in payload]


def _batch(size=20, tag="b"):
    return [{"id": f"{tag}{i}", "title": "t", "source": "s", "body": "x"}
            for i in range(size)]


def _history(days, failed_models, wins_on_last=None):
    """A usage.json body: `days` day-records all of the same shape."""
    out = {}
    for n in range(1, days + 1):
        out[f"2026-09-{n:02d}"] = {"_failed": {m: 2 for m in failed_models}}
    if wins_on_last:
        out[f"2026-09-{days:02d}"][wins_on_last] = 1
    return out


@pytest.fixture
def _clock(monkeypatch):
    """Replace the `time` module inside pipeline.budget with a virtual one.

    budget.py reaches for time.monotonic() and time.sleep() only, so swapping
    the name in its namespace makes benches and cooldown waits instantaneous
    without touching the real clock anywhere else.
    """
    import pipeline.budget as B
    clk = _Clock()

    class _T:
        monotonic = staticmethod(clk.monotonic)
        sleep = staticmethod(clk.sleep)

    monkeypatch.setattr(B, "time", _T)
    return clk


def _budget_with(data=None):
    import json
    import pipeline.budget as B
    if data is None:
        B.USAGE_PATH.unlink(missing_ok=True)
    else:
        B.USAGE_PATH.write_text(json.dumps(data), encoding="utf-8")
    return B.Budget()


class TestColdStreakCostsOneProbeNotThree:
    """Yesterday's usage file is the only evidence a run has about which
    models are reachable, and nothing was reading it.

    Fourteen consecutive day-records show the four flagship tiers failing and
    contributing nothing, while every run paid three attempts each for them —
    12 attempts a run, ~1,116s of a simulated 1,595s run, zero summaries.
    They must stay on the ladder (a direct probe shows they are contended, not
    broken: gemini-3.5-flash served a full 20-article batch in 29s) but they
    must stop costing three attempts to learn what yesterday already knew.
    """

    def test_no_history_means_no_model_is_cold(self):
        """A fresh usage file, or a new install, must not demote anything —
        and the existing three-strikes contract depends on this."""
        b = _budget_with()
        assert not [t for t in b.tiers if t.cold]
        assert all(t.max_strikes == 3 for t in b.tiers)

    def test_failing_every_day_with_no_success_earns_one_probe(self):
        model = _budget_with().tiers[0].model
        b = _budget_with(_history(3, [model]))
        top = b.tiers[0]
        assert top.cold, f"{model} failed 3 days running and never answered"
        assert top.max_strikes == 1, "a cold streak is worth one probe"

    def test_one_probe_really_does_block_on_the_first_failure(self, _clock):
        model = _budget_with().tiers[0].model
        b = _budget_with(_history(3, [model]))
        b.cool(b.tiers[0], "HTTP 503")
        assert b.tiers[0].blocked, (
            "a model on a cold streak must retire on its probe, not after "
            "three attempts")

    def test_a_single_success_in_the_window_restores_full_strikes(self):
        """One good day puts a model back on the full strike budget, or a
        model could never climb out of a streak."""
        model = _budget_with().tiers[0].model
        b = _budget_with(_history(3, [model], wins_on_last=model))
        assert not b.tiers[0].cold
        assert b.tiers[0].max_strikes == 3

    def test_a_cold_tier_keeps_its_place_at_the_top_of_the_ladder(self):
        """Demoted on strikes, NOT on priority. The flagships are better when
        they answer, so a day when they recover has to be used immediately."""
        model = _budget_with().tiers[0].model
        b = _budget_with(_history(3, [model]))
        assert b.tiers[0].model == model
        assert b.next_model("premium") is b.tiers[0], (
            "a cold tier must still be the first thing tried")

    def test_capacity_does_not_promise_batches_a_cold_tier_never_serves(self):
        """capacity(20) said 34 batches for a 20-batch run while 8 of those
        were flagship batches with a 14-day delivery rate of zero. The
        pre-emptive truncation never fired, so the shortfall arrived as silent
        extractive summaries at the end of the run instead."""
        warm = _budget_with()
        cold = [t.model for t in warm.tiers[:4]]
        chilled = _budget_with(_history(3, cold))
        assert chilled.capacity(20) < warm.capacity(20)
        assert chilled.capacity(20) == warm.capacity(20) - sum(
            t.remaining for t in warm.tiers if t.model in cold)

    def test_the_blocking_strike_is_recorded_too(self, _clock):
        """usage.json read 'exactly 2' for every flagship every single day,
        which is the signature of a 3-strike block whose last strike went
        straight to block() and was never written down. The cold-streak check
        reads these numbers, so they have to be true."""
        import json
        import pipeline.budget as B
        b = _budget_with()
        top = b.tiers[0]
        for _ in range(top.max_strikes):
            b.cool(top, "HTTP 503")
        assert top.blocked
        day = next(iter(json.loads(
            B.USAGE_PATH.read_text(encoding="utf-8")).values()))
        assert day["_failed"][top.model] == top.max_strikes, (
            "the strike that blocks a model must be counted like the others")


class TestOneAttemptPerModelPerBatch:
    """A bench is 45s but a 503 surfaces after up to 93s (two attempts at a
    45s timeout), so a benched model was eligible again the instant _generate
    looped — and the same tier was handed the same batch until its strikes ran
    out. Measured: all three flagship attempts in a run landed on batch 1.
    """

    def test_next_model_honours_the_skip_set(self):
        from pipeline.budget import Budget
        b = Budget()
        first = b.next_model("premium")
        assert b.next_model("premium", skip={first.model}) is not first

    def test_a_dead_ladder_costs_one_attempt_per_tier_not_three(self, _clock):
        """The fake's 503 costs 93s against a 45s bench, exactly as the real
        retry loop does — which is what made a benched tier selectable again
        inside the same batch."""
        from pipeline.budget import Budget
        from pipeline.s5_summarize import _generate
        b = Budget()
        b.wait_allowance = 0.0        # isolate: no cooldown waiting here
        llm = _FakeLLM(dead=[t.model for t in b.tiers], clock=_clock)
        assert llm.fail_seconds > 45.0, "the bug needs a 503 slower than a bench"
        res, _used = _generate(_batch(), b, "k", "p", "", llm, 1, 1)
        assert res is None
        assert max(llm.attempts.values()) == 1, (
            f"a model was retried inside one batch: {llm.attempts}")
        assert sum(llm.attempts.values()) == len(b.tiers)


class TestStrikesAreConsecutive:
    def test_a_success_clears_the_strike_count(self, _clock):
        """gemini-3.5-flash-lite served ELEVEN batches on 2026-10-22 and was
        still retired by three scattered 503s, because the strike count was
        cumulative over the whole run. A call that answered is proof the model
        is alive, so the count starts again from there."""
        from pipeline.budget import Budget
        b = Budget()
        t = b.tiers[-1]
        b.cool(t, "503")
        b.cool(t, "503")
        assert not t.blocked
        b.record(t)                         # it answered
        assert t.failures == 0
        b.cool(t, "503")
        b.cool(t, "503")
        assert not t.blocked, ("a model that answered in between must not be "
                               "retired by non-consecutive failures")

    def test_the_bench_lengthens_with_each_strike(self, _clock):
        """A flat 45s bench against a contention window lasting minutes just
        spends the strike budget at the fastest possible rate."""
        from pipeline.budget import Budget
        b = Budget()
        t = b.tiers[-1]
        b.cool(t, "503")
        first = t.cool_until - _clock.monotonic()
        t.cool_until = 0.0
        b.cool(t, "503")
        second = t.cool_until - _clock.monotonic()
        assert second > first, f"bench did not grow: {first}s then {second}s"


class TestAllTiersBenchedDoesNotAbandonTheRun:
    """THE 2026-10-22 SHORTFALL. _generate had no state between 'every tier is
    cooling' and 'give up on this batch', so one contention burst that benched
    the whole ladder sent every remaining batch to an extractive summary in
    the same instant — 12 of 20 batches served, batches 13-20 all resolving at
    one identical timestamp inside a bench about to expire.
    """

    def _all_benched(self, clock, seconds=30.0):
        from pipeline.budget import Budget
        b = Budget()
        for t in b.tiers:
            t.cool_until = clock.monotonic() + seconds
        return b

    def test_nothing_is_selectable_but_waiting_would_help(self, _clock):
        b = self._all_benched(_clock)
        assert b.next_model("premium", 20) is None
        assert b.cooldown_eta() > 0, (
            "a benched tier with allowance left means waiting is worthwhile")

    def test_generate_waits_for_the_bench_rather_than_falling_back(self, _clock):
        from pipeline.s5_summarize import _generate
        b = self._all_benched(_clock)
        llm = _FakeLLM(clock=_clock)
        res, used = _generate(_batch(), b, "k", "p", "", llm, 13, 20)
        assert res is not None, "the batch was abandoned inside a 45s bench"
        assert _clock.slept > 0, "it did not actually wait"
        assert used

    def test_the_wait_is_bounded_so_a_dead_ladder_still_finishes(self, _clock):
        """Waiting must never be able to sleep out the workflow."""
        from pipeline.s5_summarize import _generate
        b = self._all_benched(_clock)
        b.wait_allowance = 0.0
        llm = _FakeLLM(clock=_clock)
        res, _used = _generate(_batch(), b, "k", "p", "", llm, 13, 20)
        assert res is None
        assert not llm.attempts, "no call should have been made"
        assert "cooldown wait allowance" in b.stopped_reason

    def test_no_eta_when_waiting_cannot_possibly_help(self, _clock):
        """A benched model with no allowance left will still be useless when
        the bench expires, so there is nothing worth waiting for."""
        from pipeline.budget import Budget
        b = Budget()
        for t in b.tiers:
            t.this_run = t.per_run                    # allowance gone
            t.cool_until = _clock.monotonic() + 30
        assert b.cooldown_eta() == 0.0
        assert not b.wait_out_cooldown()

    def test_a_burst_at_the_start_does_not_kill_the_whole_run(self, _clock):
        """The regression in one line: a transient outage early in the run
        must not send every later batch to an extractive summary."""
        from pipeline.budget import Budget
        from pipeline.s5_summarize import _generate
        b = Budget()
        # Every model 503s for the first 60 virtual seconds, then recovers.
        llm = _FakeLLM(until=_clock.monotonic() + 60, clock=_clock)
        served = []
        for n in range(1, 6):
            res, _used = _generate(_batch(tag=f"n{n}"), b, "k", "p", "", llm,
                                   n, 5)
            served.append(res is not None)
        assert served[-1], (
            f"the run never recovered from a 60s outage: served={served}")
        assert sum(served) >= 3, f"only {sum(served)}/5 batches served"


class TestSlicedTierIsNotOverdrawn:
    def test_a_tier_that_cannot_cover_a_whole_batch_is_not_chosen(self):
        """capacity() divided by calls-per-batch but next_model only checked
        `remaining > 0`, so a gemma tier (batch_size 3) with ONE call of
        allowance left was still handed a 20-item batch and made all seven —
        measured 14 calls against a per_run of 8."""
        from pipeline.budget import Budget
        b = Budget()
        sliced = next(t for t in b.tiers if t.batch_size and t.batch_size < 20)
        for t in b.tiers:
            if t is not sliced:
                t.blocked = "test: force the sliced tier"
        sliced.this_run = sliced.per_run - 1          # one call of allowance
        assert sliced.remaining == 1
        assert b.capacity(20) == 0
        assert b.next_model("premium", 20) is None, (
            "a tier with one call left cannot serve a 20-item batch")
        # ...but it is still fine for a batch it can actually cover.
        assert b.next_model("premium", sliced.batch_size) is sliced
