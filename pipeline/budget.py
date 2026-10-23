"""Per-model API budgeting and pacing.

Free-tier limits are per model and differ by more than an order of magnitude
(gemini-3.6-flash allows 20 requests a day; gemini-3.5-flash-lite allows 500),
so a single global counter cannot express the constraint. This tracks each
model separately and hands out the next usable one.

Three limits, each with a different failure meaning:

  RPD       requests per UTC day. Tracked persistently across runs, because
            the quota is per day and this project runs more than once.
  RPM       requests per minute. Enforced by sleeping, not by failing — the
            calls are wanted, they just have to be spread out.
  per-run   a self-imposed cap so one run cannot eat the whole day.

Hitting any of these is normal: the caller falls back to extractive summaries.
Only a provider 429 is treated as a stop condition.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from .config import CACHE_DIR, settings, writes_enabled
from .utils import log

USAGE_PATH = CACHE_DIR / "usage.json"

# Blocks that must survive within one process but NOT across days.
#
# s5.run(), s5.top_up() and s6b_judge.run() each construct their own Budget(),
# which re-reads usage.json. A 503 block is deliberately persist=False (a
# transient overload should not cost you the model tomorrow), but that meant
# it evaporated the moment the next stage built a new Budget — so top_up
# cheerfully retried the same dead flagship and burned four more calls.
# Measured: 4 of 11 stage-5 calls returned nothing for exactly this reason.
_RUN_BLOCKS: dict[str, str] = {}


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _load() -> dict:
    if USAGE_PATH.exists():
        try:
            loaded = json.loads(USAGE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, ValueError):
            log("budget", "! usage counter corrupt, starting fresh")
            return {}
        # Valid JSON that is not an object gets through json.loads untouched
        # — a truncated write, a hand edit, a half-committed file from an
        # interrupted run all produce a list or None. Budget.__init__ then
        # does self._data.get(_today(), {}) and dies with AttributeError,
        # which is NOT a JSONDecodeError, taking stage 5 with it after
        # stages 1-4 have spent their whole network budget. This file is
        # committed by CI every run, so a bad one is persistent, and
        # _cold_streak now reads its day history to size strike budgets.
        if not isinstance(loaded, dict):
            log("budget", "! usage counter is not an object, starting fresh")
            return {}
        return loaded
    return {}


def _calls_per_batch(tier: "ModelBudget", batch_size: int) -> int:
    """How many API calls one batch of `batch_size` costs this tier.

    The gemma tiers cap a batch at 3 items, so a 20-item batch is 7 calls to
    them and 1 to a Flash tier. Both the capacity estimate and the tier
    SELECTION have to use the same number, or they disagree: capacity()
    divided while next_model() only checked `remaining > 0`, so a gemma tier
    with one call of allowance left was still handed a whole 20-item batch
    and made all seven — measured overdraft, 14 calls against a per_run of 8.
    """
    if batch_size <= 0 or not tier.batch_size or tier.batch_size >= batch_size:
        return 1
    return -(-batch_size // tier.batch_size)      # ceil


class ModelBudget:
    """One model's allowance for today."""

    def __init__(self, spec: dict, used_today: int, reserve: float,
                 strikes: int = 3, cold: bool = False) -> None:
        self.model: str = spec["model"]
        self.rpm: int = spec.get("rpm", 5)
        self.rpd: int = spec.get("rpd", 20)
        self.per_run: int = spec.get("max_batches_per_run", 5)
        # Some models have a much lower TPM than the Gemini flash family
        # (Gemma is 16K against 250K), so a batch sized for Flash would blow
        # their tokens-per-minute ceiling. None means "use the global size".
        self.batch_size: int | None = spec.get("batch_size")

        # Hold a slice of the daily quota back so an ad-hoc `--test-llm` or a
        # second run later in the day is not locked out by the first.
        self.usable_rpd: int = max(1, int(self.rpd * (1 - reserve)))

        self.used_today: int = used_today
        self.this_run: int = 0
        self.blocked: str = ""          # set when the provider itself refuses
        # A 5xx means "busy right now", not "broken". Blacklisting a model for
        # the whole run on one of those is why the four best models on the
        # ladder recorded zero successful calls across ten days: the first
        # batch 503'd on each of them in turn and every later batch went
        # straight to the cheap tiers, even though the flagship was fine
        # thirty seconds later.
        self.cool_until: float = 0.0
        self.failures: int = 0          # transient failures seen this run

        # How many transient failures this model is allowed in one run before
        # it is retired. Three for a model that has answered recently; one for
        # a model on a cold streak (see Budget._cold_streak).
        #
        # Measured over 14 consecutive day-records in usage.json: the four
        # flagship tiers were attempted THREE times each, every single run,
        # and delivered zero summaries — 12 attempts a run for nothing. The
        # usage file only ever showed two, because the third strike went
        # straight to block() and was never recorded. A simulation of the real
        # ladder put ~1,116s of a 1,595s run inside those failed attempts.
        #
        # They stay first on the ladder, because a direct probe confirms they
        # are contended rather than broken (gemini-3.5-flash served a full
        # 20-article batch in 29s), so a good day should still be used. One
        # probe keeps that upside and drops 8 of the 12 wasted attempts.
        self.max_strikes: int = max(1, strikes)
        self.cold: bool = cold

    @property
    def cooling(self) -> bool:
        return time.monotonic() < self.cool_until

    @property
    def allowance(self) -> int:
        """Calls left today ignoring benches and blocks.

        `remaining` reports 0 while a model is cooling, which is right for
        selection and wrong for deciding whether waiting out the bench could
        help — a model with no allowance left will still be useless in 45s.
        """
        return max(0, min(self.usable_rpd - self.used_today,
                          self.per_run - self.this_run))

    @property
    def remaining(self) -> int:
        if self.blocked or self.cooling:
            return 0
        return self.allowance

    @property
    def min_interval(self) -> float:
        """Seconds between calls to stay inside RPM, with a little slack."""
        return 60.0 / max(1, self.rpm) * 1.1

    def __str__(self) -> str:
        return (f"{self.model} {self.used_today}/{self.usable_rpd} today "
                f"(cap {self.rpd}), {self.remaining} left"
                + (" [cold streak: probe only]" if self.cold else ""))


class Budget:
    """The full tier ladder, plus pacing."""

    def __init__(self) -> None:
        cfg = settings()["llm"]
        bcfg = cfg.get("budget", {})
        self.enabled: bool = bcfg.get("enabled", True)
        self.pace: bool = bcfg.get("pace", True)
        reserve: float = bcfg.get("reserve_fraction", 0.15)
        self.strikes: int = int(bcfg.get("strikes", 3))
        self.cold_strikes: int = int(bcfg.get("cold_strikes", 1))
        self.cold_days: int = int(bcfg.get("cold_streak_days", 3))
        self.wait_allowance: float = float(
            bcfg.get("cooldown_wait_seconds", 240.0))
        self._waited: float = 0.0

        self._data = _load()
        day = self._data.get(_today(), {})
        blocked_today = set(day.get("_blocked", []))
        cold = self._cold_streak()
        self.tiers = []
        for spec in cfg["tiers"]:
            model = spec["model"]
            tier = ModelBudget(
                spec, day.get(model, 0), reserve,
                strikes=self.cold_strikes if model in cold else self.strikes,
                cold=model in cold)
            if tier.model in _RUN_BLOCKS:
                tier.blocked = _RUN_BLOCKS[tier.model]
            elif tier.model in blocked_today:
                # Refused by the provider earlier today. Retrying costs four
                # attempts with exponential backoff before failing the same way.
                tier.blocked = "quota refused earlier today"
            self.tiers.append(tier)
        if blocked_today:
            log("budget", f"skipping {len(blocked_today)} model(s) already "
                          f"refused today: {', '.join(sorted(blocked_today))}")
        if cold:
            log("budget", f"{len(cold)} model(s) on a cold streak — one probe "
                          f"each, not {self.strikes}: {', '.join(sorted(cold))}")
        self._last_call: dict[str, float] = {}
        self.stopped_reason: str = ""

    def _cold_streak(self) -> set[str]:
        """Models tried on each of the last few days that never once answered.

        The usage file is the only evidence a run has about which models are
        actually reachable, and nothing was reading it. Fourteen consecutive
        day-records show the same four flagships failing every run and
        contributing nothing, while each run paid three attempts apiece for
        them — and the day BEFORE said exactly the same thing.

        Deliberately conservative, because the flagships are better when they
        do answer and must not be dropped from the ladder:

          - it needs a full `cold_streak_days` of history, so a fresh usage
            file (or a new install) treats every model as healthy;
          - a single success anywhere in the window clears the streak, so one
            good day puts a model back on full strikes;
          - the streak only reduces the model's STRIKE budget. Its position on
            the ladder does not move, so a flagship is still the first thing
            tried and a day when it recovers is used immediately.

        The success counts are only as honest as record() (see its contract).
        A caller that debits before the call inflates them, which can clear a
        streak that was real — so the failure mode is a model keeping its full
        three strikes when it had earned one probe. That costs two extra
        attempts and never demotes a model that was working, which is the
        direction this should fail in.
        """
        days = sorted(self._data)[-self.cold_days:]
        if len(days) < self.cold_days:
            return set()
        wins: dict[str, int] = {}
        losses: dict[str, int] = {}
        for key in days:
            rec = self._data.get(key)
            if not isinstance(rec, dict):
                continue
            for model, n in rec.items():
                if not model.startswith("_") and isinstance(n, int):
                    wins[model] = wins.get(model, 0) + n
            failed = rec.get("_failed")
            if isinstance(failed, dict):
                for model, n in failed.items():
                    if isinstance(n, int):
                        losses[model] = losses.get(model, 0) + n
        # At least one recorded failure per day in the window, and not one
        # success in it.
        return {m for m, n in losses.items()
                if n >= len(days) and not wins.get(m)}

    # --- selection -------------------------------------------------------

    def next_model(self, prefer: str = "premium", batch_size: int = 0,
                   skip: "set[str] | frozenset[str]" = frozenset()
                   ) -> ModelBudget | None:
        """Pick a model for the next call.

        `prefer` routes by what the work is worth, not just by batch order:

          premium  the strongest model with quota. For stories a person will
                   actually read at the top of the page.
          bulk     the cheapest model with quota, searched from the bottom.
                   For low-frequency material — social posts, repository
                   listings, minor items — where a weaker summary is fine and
                   the flagship's 20-a-day allowance would be wasted.

        Both fall back through the whole ladder, so "bulk" still gets served
        by a flagship model if the cheap tiers are exhausted, and vice versa.

        `batch_size` is the size of the batch about to be sent. A tier that
        slices the batch needs that many CALLS of allowance, not one, and
        checking only `remaining > 0` meant a gemma tier with a single call
        left still took a 20-item batch and made all seven — 14 calls against
        a per_run of 8. Zero keeps the old "any allowance will do" meaning.

        `skip` names models already tried for THIS batch. The bench is 45s but
        a 503 can take 93s to come back (2 attempts at a 45s timeout), so a
        benched model was already eligible again by the time the caller looped
        — and _generate re-picked the model it had just benched, inside the
        same batch, until all three strikes were gone. Measured: 3 attempts
        per flagship per run, all of them on batch 1. The point of the ladder
        is that a DIFFERENT model is likelier to answer, so the same one must
        not be offered twice for one batch.
        """
        # `enabled: false` disables the QUOTA accounting, not the tier ladder.
        # Returning tiers[0] unconditionally ignored `blocked`, so a caller
        # looping until next_model() returns None — which _generate does —
        # would spin forever against a permanently failing model.
        order = [t for t in (self.tiers if prefer == "premium"
                             else reversed(self.tiers)) if t.model not in skip]
        if not self.enabled:
            for tier in order:
                if not tier.blocked:
                    return tier
            self.stopped_reason = "every model has failed this run"
            return None
        for tier in order:
            if tier.remaining >= _calls_per_batch(tier, batch_size):
                return tier

        self.stopped_reason = ("all model quotas exhausted for today — "
                               "remaining items use extractive summaries")
        return None

    # --- cooldowns -------------------------------------------------------

    def cooldown_eta(self) -> float:
        """Seconds until the earliest benched model with allowance returns.

        0.0 when waiting cannot help: nothing is cooling, or the models that
        are have no allowance left anyway so they will still be useless when
        the bench expires.
        """
        now = time.monotonic()
        etas = [t.cool_until - now for t in self.tiers
                if not t.blocked and t.cooling and t.allowance > 0]
        return max(0.0, min(etas)) if etas else 0.0

    def wait_out_cooldown(self) -> bool:
        """Sleep until the next benched model comes back. True if it waited.

        THE MEASURED SHORTFALL. A bench is 45 seconds and a run is tens of
        minutes, but _generate had no state between "every tier is cooling"
        and "give up on this batch". So the instant a contention burst benched
        the whole ladder at once, next_model() returned None and EVERY
        remaining batch fell to an extractive summary — all of them resolved
        inside the same 45-second window the bench was about to end.
        Reproduced against the real ladder: capacity(20) promised 34 batches,
        12 were served, and batches 13-20 all died at one identical
        timestamp. That is the 2026-10-22 run exactly (calls=12 against the 20
        a 400-item budget needs).

        Bounded by llm.budget.cooldown_wait_seconds, so a genuinely dead
        ladder still terminates rather than sleeping out the workflow.
        """
        eta = self.cooldown_eta()
        if eta <= 0:
            return False
        left = self.wait_allowance - self._waited
        if left <= 0:
            self.stopped_reason = (
                f"every model is benched and the {self.wait_allowance:.0f}s "
                f"cooldown wait allowance is spent — remaining items use "
                f"extractive summaries")
            return False
        # Half a second past the bench, so `cooling` is certainly false after.
        nap = min(eta + 0.5, left)
        log("budget", f"all tiers benched — waiting {nap:.0f}s for the next "
                      f"one ({self._waited:.0f}s of {self.wait_allowance:.0f}s "
                      f"already spent waiting)")
        time.sleep(nap)
        self._waited += nap
        return True

    def capacity(self, batch_size: int = 0) -> int:
        """How many batches of `batch_size` the ladder can really serve.

        `remaining` counts a tier's remaining CALLS. That equals batches only
        while the tier takes a whole batch in one call. The gemma tiers have
        batch_size 3, so _generate slices a 20-item batch into 7 calls and
        debits 7 — meaning 8 remaining calls is about ONE batch, not eight.

        Counting calls as batches overstated the ladder by ~14 batches. At a
        summarisation budget of 150 the slack hid it; at 400 the run needs 20
        batches, so if the flagship and flash-lite tiers are spent the caller
        would believe it had capacity it does not have, skip its pre-emptive
        truncation, and instead discover the shortfall one failed batch at a
        time late in the run.

        Called with no argument it keeps the old call-counting meaning.
        """
        if not self.enabled:
            return 10 ** 6
        if batch_size <= 0:
            return sum(t.remaining for t in self.tiers)
        total = 0
        for t in self.tiers:
            if t.cold:
                # A tier on a cold streak has delivered nothing for days. It
                # is still TRIED — one probe, first on the ladder — but
                # PROMISING its batches is how a run that needed 20 batches
                # convinced itself it had 34: eight of those were flagship
                # batches with a measured 14-day delivery rate of zero. The
                # pre-emptive truncation then never fired and the shortfall
                # arrived as silent extractive summaries at the end of the
                # run instead. Count only what the ladder reliably serves.
                continue
            total += t.remaining // _calls_per_batch(t, batch_size)
        return total

    # --- execution -------------------------------------------------------

    def wait(self, tier: ModelBudget) -> None:
        """Sleep if needed so this model's RPM is not exceeded."""
        if not self.pace:
            return
        last = self._last_call.get(tier.model)
        if last is None:
            return
        gap = tier.min_interval - (time.monotonic() - last)
        if gap > 0:
            log("budget", f"pacing {gap:.0f}s for {tier.model} (RPM {tier.rpm})")
            time.sleep(gap)

    def record(self, tier: ModelBudget) -> None:
        """Count a call that SUCCEEDED. Never call this before the call.

        That is a contract, not a style note, and two things now depend on it:

          - the per-model day counter, which _cold_streak() reads as evidence
            that a model is reachable at all;
          - the strike reset below, which treats the call as proof of life.

        Debit-then-call makes both of those lie in the same direction: a model
        that 503s on every batch is written down as having answered. Caught
        once already in _generate (a 429 or 503 returns nothing but still
        spent the local allowance, so the counter drifted pessimistic against
        the provider); the isolated single-success rows against the gemma
        tiers in data/cache/usage.json — 2026-10-11 and 2026-10-20, one
        success apiece beside two recorded failures — are what that looks like
        from the outside, and the judge stage selects exactly those tiers.
        """
        tier.this_run += 1
        tier.used_today += 1
        # CONSECUTIVE failures, not failures all run. `failures` was cumulative,
        # so gemini-3.5-flash-lite — which served 11 batches on 2026-10-22 —
        # was still retired by three scattered 503s. A call that answered is
        # proof the model is alive; the strike count starts again from there.
        tier.failures = 0
        tier.cool_until = 0.0
        self._last_call[tier.model] = time.monotonic()

        day = self._data.setdefault(_today(), {})
        day[tier.model] = tier.used_today          # "_blocked" key is preserved
        for old in sorted(self._data)[:-14]:       # keep a fortnight
            self._data.pop(old, None)
        if writes_enabled():
            USAGE_PATH.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def cool(self, tier: ModelBudget, why: str, seconds: float = 45.0) -> None:
        """Bench a model briefly instead of retiring it for the run.

        Overload is a property of the moment, not of the model. Enough strikes
        in one run does start to look like the model, so at that point it is
        blocked properly and the ladder moves on for good. How many is
        `tier.max_strikes` — three normally, one for a model that has not
        answered on any of the last few days.
        """
        tier.failures += 1
        # BEFORE the strike check, not after. The blocking strike used to
        # return early and was never recorded, which is why usage.json read
        # "exactly twice each, every day" for the four flagships while the run
        # was really attempting them three times each: a constant 2 is the
        # signature of a 3-strike block whose last strike is invisible. The
        # cold-streak check above reads these numbers, so they have to be true.
        self._record_failure(tier)
        if tier.failures >= tier.max_strikes:
            self.block(tier, f"{why} ({tier.failures} consecutive strike(s))")
            return
        # Doubling, not flat. A flat 45s bench against a contention window
        # that lasts a minute or two just spends the strike budget at the
        # fastest possible rate: 45s, then blocked. 45/90/180 rides out ~4
        # minutes of contention on a tier that is otherwise serving the run.
        seconds *= 2 ** (tier.failures - 1)
        tier.cool_until = time.monotonic() + seconds
        log("budget", f"{tier.model} busy — benched {seconds:.0f}s "
                      f"(strike {tier.failures}/{tier.max_strikes}): {why}")

    def _record_failure(self, tier: ModelBudget) -> None:
        """Count an attempt that did not succeed, in the usage file.

        record() only ever ran AFTER a successful call, so a model that was
        tried and failed every single day left no trace anywhere. Ten days of
        usage data showed nothing at all for the four best models on the
        ladder, which reads as "never selected" when the truth was "selected
        first, every run, and never once answered". A number that only counts
        wins cannot tell you which of those you are looking at.
        """
        try:
            day = self._data.setdefault(_today(), {})
            fails = day.setdefault("_failed", {})
            if isinstance(fails, dict):
                fails[tier.model] = fails.get(tier.model, 0) + 1
                if writes_enabled():
                    USAGE_PATH.write_text(json.dumps(self._data, indent=2),
                                      encoding="utf-8")
        except Exception:
            pass          # accounting must never break a run

    def block(self, tier: ModelBudget, why: str, persist: bool = False) -> None:
        """Stop using this model; fall through to the next tier.

        `persist` writes the block into the day's usage file. Reserve it for
        quota refusals (429), which last until the quota resets, and never
        use it for transient failures (503 overload), which do not.

        This matters because the local counter is only a self-limit: it
        cannot see calls made before it existed, or by anything else sharing
        the key. On a live run gemini-3.6-flash returned 429 after just 4
        recorded calls, because ~16 earlier calls were invisible to it. The
        provider's refusal is authoritative; the counter is a guess.
        """
        tier.blocked = why
        _RUN_BLOCKS[tier.model] = why      # survives the next Budget() in this run
        log("budget", f"! {tier.model} blocked: {why}")
        if persist:
            day = self._data.setdefault(_today(), {})
            day.setdefault("_blocked", []).append(tier.model)
            if writes_enabled():
                USAGE_PATH.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
            log("budget", f"  {tier.model} stays blocked for the rest of the UTC day")

    # --- reporting -------------------------------------------------------

    def report(self) -> str:
        return " | ".join(str(t) for t in self.tiers)
