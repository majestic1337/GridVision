from ingest.enrichment import TextNormalizer, QualityScorer
from ingest.processor import paragraph_chunk_text


def test_normalize_removes_hyphen_breaks():
    text = "connec-\n tion  \nnext"
    assert TextNormalizer.normalize(text) == "connection next"


def test_normalize_keep_paragraphs_preserves_breaks():
    text = "line 1  \n\nline 2"
    assert TextNormalizer.normalize_keep_paragraphs(text) == "line 1\n\nline 2"


def test_quality_scorer_boosts_tech_terms():
    score, reasons = QualityScorer.evaluate("Check wire 130A16 at J1 connector.", "text")
    assert score > 0.5
    assert any("tech_terms_found" in r for r in reasons)


def test_quality_scorer_penalizes_short_text():
    score, reasons = QualityScorer.evaluate("short text", "text")
    assert score < 0.5
    assert any("low_word_count" in r for r in reasons)


def test_paragraph_chunk_text_groups_small_paras():
    text = "one two\n\nthree four\n\nfive six"
    chunks = paragraph_chunk_text(text, min_words=3, max_chars=100)
    assert len(chunks) == 2
    assert "one two" in chunks[0]
    assert "three four" in chunks[0]
