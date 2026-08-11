#!/usr/bin/env python3
"""Regression tests. Stdlib only, like the thing it tests: python3 scripts/test_pdf2md.py

Kept deliberately small -- this covers the one bug that has actually bitten, where a
1173-page document OCR'd for half an hour and then died in the merge step with an error
naming neither the function nor the page.
"""
import random
import sys
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
