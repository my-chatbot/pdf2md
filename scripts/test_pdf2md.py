#!/usr/bin/env python3
"""Regression tests. Stdlib only, like the thing it tests: python3 scripts/test_pdf2md.py

Kept deliberately small -- this covers the one bug that has actually bitten, where a
1173-page document OCR'd for half an hour and then died in the merge step with an error
naming neither the function nor the page.
"""
import json
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pdf2md as m

HDR = "www.lawcommission.gov.np"
BODY = f"{HDR}\nbody one\nbody two\nbody three\n5"


def old_walk(pages):
    """strip_noise as it was before the fix, minus the crash: the fixed index tuple,
    skipping entries it had already blanked. Reference for proving the deduplicated
    index set selects the same lines, rather than merely not raising."""
    lines = [[l.strip() for l in p.splitlines() if l.strip()] for p in pages]
    edges = Counter()
    for L in lines:
        edges.update({L[0], L[-1]} if L else set())
    cutoff = max(2, len(lines) // 2)
    repeated = {t for t, c in edges.items() if c >= cutoff}
    out = []
    for L in lines:
        keep = list(L)
        for idx in (0, 1, -1, -2):
            if not keep or len(keep) <= abs(idx):
                continue
            t = keep[idx]
            if t is None:                       # the guard the shipped code lacked
                continue
            if t in repeated or m.NUMBERISH.fullmatch(t):
                keep[idx] = None
        out.append("\n".join(t for t in keep if t is not None))
    return out


def test_header_plus_page_number():
    """The reported failure: a page with only a running header and a page number.
    Both get blanked, then index -2 wraps back onto index 0 and reads the None."""
    pages = [BODY] * 6 + [f"{HDR}\n7"]
    assert m.strip_noise(pages)[-1] == "", "a header+number page should empty out"


def test_short_pages_of_every_shape():
    """Every page length where the top and bottom windows can overlap."""
    for n in range(0, 6):
        for filler in (HDR, "7", "real text"):
            page = "\n".join([HDR] + [filler] * n)
            m.strip_noise([BODY] * 6 + [page])          # must not raise


def test_body_text_survives():
    """The window is deliberately narrow -- a clause marker like '१०.' carries text on
    the same line, so it must never be mistaken for a page number."""
    pages = [BODY] * 6 + [f"{HDR}\n१०. यो दफा हो\nमध्य पङ्क्ति\nअन्तिम पङ्क्ति\n12"]
    kept = m.strip_noise(pages)[-1]
    assert "१०. यो दफा हो" in kept, "clause text was dropped"
    assert HDR not in kept, "running header survived"
    assert not kept.endswith("12"), "page number survived"


def test_matches_the_old_behaviour():
    """Fuzz: the fix must select the same lines as the original index walk, not just
    avoid raising."""
    random.seed(1)
    vocab = [HDR, "५", "7", "१०. text", "body", "।", "- -", "2024", ""]
    for _ in range(4000):
        pages = ["\n".join(random.choice(vocab) for _ in range(random.randint(0, 7)))
                 for _ in range(random.randint(1, 9))]
        assert m.strip_noise(pages) == old_walk(pages), pages


def test_parse_pages():
    """The spec forms the workflow and the shard planner depend on."""
    assert m.parse_pages("5", 183) == [5]
    assert m.parse_pages("169-173", 183) == [169, 170, 171, 172, 173]
    assert m.parse_pages("1-183/10", 183) == list(range(1, 184, 10))
    assert len(m.parse_pages("1-183:10", 183)) == 10
    assert m.parse_pages("1-10,21-30", 183) == list(range(1, 11)) + list(range(21, 31))
    assert m.parse_pages(None, 4) == [1, 2, 3, 4]


def test_chunk_planner_covers_everything_once():
    counts = [(Path(f"d{i}"), p) for i, p in enumerate([1, 2, 30, 183, 389, 1173])]
    seen = Counter()
    for group in m.plan_chunks(counts, 10, 10):
        for pdf, a, b in group:
            seen.update((pdf, p) for p in range(a, b + 1))
    assert all(seen[(pdf, p)] == 1 for pdf, n in counts for p in range(1, n + 1))
    assert sum(seen.values()) == sum(n for _, n in counts)


def _corpus(tmp, sidecar, pdf_name):
    """A throwaway document/acts/ holding one PDF and its sidecar."""
    d = Path(tmp) / "document" / "acts"
    d.mkdir(parents=True)
    (d / m.SIDECAR).write_text(json.dumps(sidecar, ensure_ascii=False), encoding="utf-8")
    (d / pdf_name).touch()
    return Path(tmp) / "document", d / pdf_name


def test_frontmatter_matches_the_requested_shape():
    """The exact block asked for, key order included -- normalised keys first, so
    every document in the corpus filters the same way."""
    name = "12171_foo.pdf"
    with tempfile.TemporaryDirectory() as tmp:
        root, pdf = _corpus(tmp, {
            "_meta": {"category_label": "ऐन"},
            "12171": {"id": "12171", "url": "https://example.np/12171/",
                      "title": "विदेशी लगानी", "pdf_path": name}}, name)
        got = m.frontmatter(pdf, root, m.load_sidecar(pdf.parent))
    assert got == ('---\n'
                   'category: "acts"\n'
                   'category_label: "ऐन"\n'
                   'id: "12171"\n'
                   'url: "https://example.np/12171/"\n'
                   'title: "विदेशी लगानी"\n'
                   f'pdf_path: "acts/{name}"\n'
                   '---\n\n'), got


def test_frontmatter_normalises_the_other_vocabulary():
    """The bulletins spell it cumulative/pdf_url and carry no title; the same keys
    must come out, with the native fields kept underneath and never duplicated."""
    name = "804_x.pdf"
    with tempfile.TemporaryDirectory() as tmp:
        root, pdf = _corpus(tmp, {
            "_meta": {"title_template": "वर्ष {volume}, अङ्क {issue}"},
            "804": {"cumulative": "804", "pdf_url": "https://example.np/a.pdf",
                    "volume": "34", "issue": "18", "serial": "1", "pdf_path": name}}, name)
        got = m.frontmatter(pdf, root, m.load_sidecar(pdf.parent))
    assert 'id: "804"' in got and 'url: "https://example.np/a.pdf"' in got, got
    assert 'title: "वर्ष 34, अङ्क 18"' in got, got
    assert 'volume: "34"' in got and 'serial: "1"' in got, got
    assert "cumulative:" not in got and "pdf_url:" not in got, got


def test_frontmatter_absent_rather_than_wrong():
    """No sidecar, an unlisted document, or a corrupt file must all mean 'no
    frontmatter' -- metadata is a nicety, OCR is the job."""
    name = "a.pdf"
    with tempfile.TemporaryDirectory() as tmp:
        root, pdf = _corpus(tmp, {"1": {"pdf_path": "other.pdf"}}, name)
        assert m.frontmatter(pdf, root, m.load_sidecar(pdf.parent)) == ""
        (pdf.parent / m.SIDECAR).write_text("{ not json", encoding="utf-8")
        assert m.load_sidecar(pdf.parent) == {}
        (pdf.parent / m.SIDECAR).unlink()
        assert m.load_sidecar(pdf.parent) == {}


def test_frontmatter_title_falls_back_before_it_invents():
    """A template naming a field the entry lacks must not raise, and must not emit
    a half-substituted title."""
    name = "z.pdf"
    with tempfile.TemporaryDirectory() as tmp:
        root, pdf = _corpus(tmp, {
            "_meta": {"title_template": "{nope} {alsonope}"},
            "1": {"id": "1", "pdf_path": name}}, name)
        got = m.frontmatter(pdf, root, m.load_sidecar(pdf.parent))
    assert 'title: "z"' in got, got


def test_without_frontmatter_leaves_prose_alone():
    """Only a block at the very start counts. A --- rule inside the prose, or a
    document that never had frontmatter, must come back untouched."""
    body = "<!-- page 1 -->\n\nfirst\n\n---\n\nsecond\n"
    assert m.without_frontmatter(body) == body                  # no leading block
    assert m.without_frontmatter("---\na: \"1\"\n---\n\n" + body) == body
    assert m.without_frontmatter("---\nunterminated\n") == "---\nunterminated\n"


def test_refresh_frontmatter_is_idempotent():
    """Rewriting twice must not stack blocks or drift -- this runs over a corpus
    that already cost 26 runner-hours, so it has to be safe to repeat."""
    name = "12171_foo.pdf"
    body = "<!-- page 1 -->\n\nprose\n\n---\n\nmore prose\n"
    with tempfile.TemporaryDirectory() as tmp:
        root, pdf = _corpus(tmp, {
            "_meta": {"category_label": "ऐन"},
            "12171": {"id": "12171", "title": "t", "pdf_path": name}}, name)
        out = Path(tmp) / "out"
        (out / "acts").mkdir(parents=True)
        md = out / "acts" / "12171_foo.md"
        md.write_text(body, encoding="utf-8")

        class A:
            indir, frontmatter = str(root), True
        assert m.refresh_frontmatter(pdf, out, "acts/12171_foo", A) == 1
        first = md.read_text(encoding="utf-8")
        assert first.startswith('---\ncategory: "acts"\n')
        assert first.endswith(body)                       # prose survives verbatim
        # second run changes nothing and reports so
        assert m.refresh_frontmatter(pdf, out, "acts/12171_foo", A) == 0
        assert md.read_text(encoding="utf-8") == first
        assert first.count("\ncategory:") == 1            # no stacked blocks


def test_yaml_values_are_escaped():
    """A quote or backslash in a title would otherwise produce invalid YAML."""
    assert m.yaml_value('a "b" \\ c') == '"a \\"b\\" \\\\ c"'
    assert m.yaml_value("देवनागरी") == '"देवनागरी"'
    assert m.yaml_value(None) == '""'


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok    {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {str(e)[:90]}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
