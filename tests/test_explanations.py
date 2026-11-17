"""The card's explanation must explain the story, not echo the headline or
describe the model's input. Strings are real ones from the 17 Nov page."""
from pipeline.models import Item
from pipeline.s5_summarize import _apply, _heuristic
from pipeline.utils import echoes_title, strip_meta

T = "Manufacturing hit by surge in ransomware attacks"


def _item(title=T):
    return Item(cluster_key="k", title=title, category="cyber_attacks")


# --- headline echoes ---------------------------------------------------------
def test_headline_plus_outlet_is_an_echo():
    assert echoes_title(T + " Food & Beverage Industry News", T)


def test_real_lede_is_not_an_echo():
    lede = ("Manufacturing was hit by a surge in ransomware attacks in Q3, with "
            "Dragos counting 300 incidents across 40 countries and LockBit "
            "successors behind most of them.")
    assert not echoes_title(lede, T)


def test_cut_social_title_is_completed_not_echoed():
    title = "vm2's own repo lists its next sandbox escapes: a deferred\u2026"
    post = "vm2's own repo lists its next sandbox escapes: a deferred fix."
    assert not echoes_title(post, title)


def test_extract_drops_the_echo_sentence():
    it = _item()
    _heuristic(it, T + " Food & Beverage Industry News")
    assert it.summary == ""


def test_extract_keeps_real_sentences_after_the_echo():
    body = (T + " Food & Beverage Industry News. Dragos counted 300 incidents "
            "in the quarter, most of them against food processors in Europe.")
    it = _item()
    _heuristic(it, body)
    assert it.summary.startswith("Dragos counted 300 incidents")


# --- model describing its input ----------------------------------------------
META = [
    "The article is unavailable.",
    "This article is truncated and contains only a headline and source name "
    "with no substantive body text provided.",
    "The article is truncated and merely displays a cookie wall and "
    "page-loading notice, failing to provide details.",
    "Because the content lacks actionable facts or an explicit body, it "
    "functions merely as a headline stub.",
    "Consequently, no operational or strategic takeaways can be derived.",
]


def test_meta_sentences_are_removed():
    for s in META:
        assert strip_meta(s) == "", s


def test_real_sentence_next_to_meta_survives():
    s = ("A report from Moody's indicates that global banks are entering 2027 "
         "in a strong financial position. The article is truncated.")
    assert strip_meta(s) == ("A report from Moody's indicates that global "
                             "banks are entering 2027 in a strong financial "
                             "position.")


def test_story_about_an_article_is_not_meta():
    s = "The article's authors found 40 exposed servers in the scan."
    assert strip_meta(s) == s


def test_apply_never_publishes_a_meta_summary():
    it = _item("Revolut unit placed under judicial supervision")
    _apply(it, {"summary": META[1], "bottom_line": META[4]}, "test")
    assert it.summary == "" and it.bottom_line == ""
    assert it.summary_source != "model"


def test_apply_empty_answer_keeps_existing_extract():
    it = _item()
    _heuristic(it, "Dragos counted 300 incidents in the quarter, most of them "
                   "against food processors in Europe.")
    before = it.summary
    _apply(it, {"summary": META[0]}, "test")
    assert it.summary == before and it.summary_source == "extract"
