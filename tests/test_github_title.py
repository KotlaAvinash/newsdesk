"""GitHub card titles must never stop mid-word (regression: desc[:110])."""
from pipeline.s1b_github import _to_article

DESC = ("Open-source agent for collecting endpoint telemetry, threat hunting, "
        "network monitoring, and infrastructure inventory across large fleets")


def _repo(desc):
    return {"full_name": "acme/hunter", "description": desc,
            "html_url": "https://github.com/acme/hunter",
            "stargazers_count": 42, "created_at": "2026-11-17T00:00:00Z"}


def test_long_description_cut_on_a_whole_word():
    title = _to_article(_repo(DESC), "cyber", "new").title
    tail = title.split(" — ", 1)[1].rstrip("\u2026")
    assert DESC.startswith(tail)
    # the cut must land on a word boundary, never inside "infrastructure"
    assert DESC[len(tail)] == " "
    assert title.endswith("\u2026")


def test_short_description_kept_whole():
    assert _to_article(_repo("Tiny tool."), "cyber", "new").title == \
        "acme/hunter — Tiny tool."


def test_no_description_uses_repo_name():
    assert _to_article(_repo(None), "cyber", "new").title == "acme/hunter"
