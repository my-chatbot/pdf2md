#!/usr/bin/env python3
"""Nepali PDF -> Markdown via Tesseract OCR, shaped for a RAG pipeline.

Pages with a genuine Unicode text layer are read straight from the PDF with
pdftotext; everything else (scanned, or legacy-font Preeti/Kantipur, whose text
layer is ASCII bytes mapped through a non-Unicode font) is rasterized and OCR'd.
Running headers, footers and page numbers are stripped so they don't end up in
every retrieved chunk. One .md per document, nothing else.

With --shard and --total-shards, the script OCRs only its own share of the work,
split at --chunk page granularity so N runners get an even load however lopsided
the corpus is. Each shard leaves its pages in the per-document cache; once every
cache is overlaid in one place, a single --merge-only pass assembles them into
one .md per document.
"""
import argparse
import concurrent.futures as cf
import difflib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

# only digits (ASCII or Devanagari), danda, pipe, dot, space -- a page-number line
NUMBERISH = re.compile(r"[\d०-९।|.\s]+")

# a genuine Unicode text layer is mostly Devanagari; legacy-font garbage is ASCII
DEVANAGARI = re.compile(r"[\u0900-\u097F\u200C\u200D]")

# minimum character agreement between the text layer and OCR of page 1 before
# the layer is trusted for the whole document. sambidhan.pdf's layer is
# Devanagari but ~40% wrong (legacy-font derived), so ratio alone is not enough.
LAYER_AGREEMENT = 0.9

# mean LSTM word confidence above which a fast (low-dpi) pass is kept as-is
FAST_CONF = 90.0

# fraction of words whose re-encoding matches the page's own text layer before
# a page counts as certified (the rest gets a higher-dpi re-OCR)
CERT_MIN = 0.9

# fraction of the calibration sample's OCR words the encoder must cover before
# certification means anything; below this the codec is too thin to trust
CODEC_COVERAGE = 0.5


def usable_cpus() -> int:
    """Cores this process may actually run on. os.sched_getaffinity is Linux-only,
    so fall back to the whole machine elsewhere."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def agreement(a: str, b: str) -> float:
    """Character-level agreement after whitespace is stripped (0.0-1.0)."""
    a = "".join(a.split()).replace("\f", "")
    b = "".join(b.split()).replace("\f", "")
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def learn_rules(pairs: list[tuple[str, str]]) -> tuple[dict[str, str], set[str]]:
    """Align each sample page's text layer against its OCR word by word and
    collect the layer's corruption rules (corrupted layer word -> what OCR
    reads). Returns (rules, seen): rules maps a layer word to its correction,
    seen holds every layer word that aligned 1:1 with itself -- words proven
    genuine by the sample. Whole-word rules are deliberate: char-level
    alignment fragments corruptions (मममि -> 'मम' + 'मि') into rules that
    misfire on genuine text. Only words that mapped to a correction at least
    twice are trusted -- one-off alignment noise is dropped."""
    rules: Counter = Counter()
    seen: set[str] = set()
    for layer, ref in pairs:
        a = layer.replace("\f", "").split()
        b = ref.replace("\f", "").split()
        sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "equal":
                seen.update(a[i1:i2])
            elif i2 - i1 == 1 and j2 - j1 == 1:   # 1:1 word pair
                lw, rw = a[i1], b[j1]
                if 1 <= len(lw) <= 40 and lw.isprintable() and rw.isprintable():
                    rules[(lw, rw)] += 1
    best: dict[str, tuple[str, int]] = {}
    for (lw, rw), c in rules.items():
        if c >= 2 and (lw not in best or c > best[lw][1]):
            best[lw] = (rw, c)
    return {w: r for w, (r, _) in best.items()}, seen


def decode_layer(text: str, rules: dict[str, str], seen: set[str]) -> tuple[str, int, int]:
    """Rewrite the text layer page word by word. Returns (decoded, known,
    novel): known = chars in words rewritten by a learned rule, novel = words
    neither rewritten nor proven genuine by the sample."""
    words = text.replace("\f", "").split()
    out: list[str] = []
    known = novel = 0
    for w in words:
        if w in rules:
            out.append(rules[w])
            known += len(w)
        else:
            if w not in seen:
                novel += 1
            out.append(w)
    return " ".join(out), known, novel


def usable(text: str) -> bool:
    """True if the text layer is real Nepali, not legacy-font garbage or noise."""
    chars = [c for c in text.replace("\f", "") if not c.isspace()]
    if len(chars) < 8:
        return False
    # ponytail: 25% fit this corpus; raise it if a digital doc is English-heavy
    return sum(1 for c in chars if DEVANAGARI.match(c)) / len(chars) >= 0.25


def parse_pages(s: str, total: int) -> list[int]:
    """'5' -> [5]; '169-173' -> [169..173]; '1-183/10' -> every 10th page of the
    range; '1-183:10' -> exactly 10 pages spread evenly; open-ended samples
    ('5/10', '5:10') run to the last page; None -> all. Comma-joins any of those
    ('1-10,101-110'), which is how a shard names its disjoint runs of one
    document. 1-based."""
    if not s:
        return list(range(1, total + 1))
    if "," in s:
        return sorted({p for part in s.split(",") for p in parse_pages(part, total)})
    spec, sep, tail = s.partition(":")
    if sep:                                 # fixed count: dynamic stride
        if not tail:
            raise ValueError(f"bad count in --pages {s!r}")
        count = int(tail)
        if count < 1:
            raise ValueError(f"bad count in --pages {s!r}")
        lo, hi = _span(spec, total, True)
        span = hi - lo + 1
        if span < 1:
            raise ValueError(f"--pages {s!r} starts past the end ({total} pages)")
        count = min(count, span)
        stride = -(-span // count)          # ceiling division: the stride is
        if lo + (count - 1) * stride > hi:  # derived, so a ceiling that
            stride -= 1                     # overshoots the tail steps down
        return [lo + i * stride for i in range(count)]
    spec, sep, tail = s.partition("/")
    stride = int(tail) if sep else 1
    if stride < 1:
        raise ValueError(f"bad stride in --pages {s!r}")
    lo, hi = _span(spec, total, stride > 1)
    if hi < lo:
        raise ValueError(f"--pages {s!r} starts past the end ({total} pages)")
    count = -(-(hi - lo + 1) // stride)     # ceiling division
    return [lo + i * stride for i in range(count)]


def _span(spec: str, total: int, sampling: bool) -> tuple[int, int]:
    """Resolve 'start[-end]' against the document's page count; an open-ended
    spec during sampling runs to the last page."""
    start, _, end = spec.partition("-")
    lo = max(1, int(start))
    hi = total if (sampling and not end) else min(total, int(end) if end else lo)
    return lo, hi


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    # Tesseract multithreads via OpenMP and grabs ~1.5 cores per process, which
    # oversubscribes the box on top of our own pool. One OMP thread per process,
    # N processes, is both faster overall and predictable.
    env = {**os.environ, "OMP_THREAD_LIMIT": "1"}
    return subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)


def page_count(pdf: Path) -> int:
    out = run(["pdfinfo", str(pdf)]).stdout
    return int(re.search(r"^Pages:\s+(\d+)", out, re.M).group(1))


def render(pdf: Path, pno: int, workdir: Path, dpi: int) -> Path:
    run(["pdftoppm", "-f", str(pno), "-l", str(pno), "-r", str(dpi), "-png", "-gray",
         str(pdf), str(workdir / "p")])
    pngs = list(workdir.glob("p-*.png"))
    if not pngs:
        raise RuntimeError(f"page {pno} did not render")
    return pngs[0]


def text_layer(pdf: Path, pno: int) -> str:
    return run(["pdftotext", "-f", str(pno), "-l", str(pno), str(pdf), "-"]).stdout


def ocr(png: Path, lang: str, psm: str) -> str:
    # ponytail: no deskew/binarize/denoise. If scans are poor, pipe through
    # `convert in.png -deskew 40% -threshold 60% out.png` first, or try --psm 3.
    return run(["tesseract", str(png), "stdout", "-l", lang, "--psm", psm]).stdout


def ocr_fast(png: Path, lang: str, psm: str) -> tuple[str, float]:
    """OCR with TSV output: the page text rebuilt from layout plus the mean
    word confidence. One tesseract call gives both, so the fast pass costs no
    extra subprocess."""
    tsv = run(["tesseract", str(png), "stdout", "-l", lang, "--psm", psm,
               "tsv"]).stdout
    rows = [l.split("\t") for l in tsv.splitlines() if l.startswith("5\t")]
    confs = [float(r[10]) for r in rows if r[10] != "-1"]
    text, prev = [], None
    for r in rows:
        key = (r[2], r[3], r[4])            # block, par, line
        if prev is not None and key[0:2] != prev[0:2]:
            text.append("\n\n")
        elif prev is not None and key[2] != prev[2]:
            text.append("\n")
        prev = key
        text.append(r[11] + " ")
    page = "".join(text).replace(" \n", "\n")
    return page, (sum(confs) / len(confs) if confs else 0.0)


def build_encoder(rules: dict[str, str], seen: set[str]) -> dict[str, str]:
    """The inverse codec: Unicode word -> the layer bytes the PDF actually
    stores for it. Identity words (seen, never observed corrupt) encode to
    themselves; words a learned rule corrected encode to their corrupt bytes,
    since that is what the document stores."""
    enc = {w: w for w in seen}
    for lw, rw in rules.items():
        enc[rw] = lw
    return enc


def certify_page(text: str, layer: str, enc: dict[str, str]) -> tuple[float, float]:
    """Re-encode the OCR'd words back into layer bytes and compare them with
    the page's actual text layer. Two independent channels (what tesseract
    read, and what the page itself encodes) agreeing certifies a word.
    Returns (certified, unknown): the fraction of words whose re-encoding
    matches the layer, and the fraction the codec cannot encode."""
    a = text.replace("\f", "").split()
    b = layer.replace("\f", "").split()
    if not a:
        return 0.0, 0.0
    encoded = [enc.get(w) for w in a]
    unknown = sum(1 for e in encoded if e is None) / len(a)
    cert = 0
    sm = difflib.SequenceMatcher(None, encoded, b, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            cert += i2 - i1
    return cert / len(a), unknown


def strip_noise(pages: list[str]) -> list[str]:
    """Drop running headers/footers and page numbers -- chunk poison for RAG.

    Two rules, both data-driven so no document-specific string is hardcoded:
      1. a first/last line repeated on most pages is a running header or footer;
      2. a numeric line at the very top or bottom is a page number (each one differs,
         so rule 1 cannot see them).
    Only the top and bottom two lines of a page are eligible, which is what keeps
    body text safe -- clause markers like '१०.' carry text on the same line.
    # ponytail: 50% and the two-line window fit this corpus. A header that appears on
    # only a third of pages needs the threshold lowered.
    """
    lines = [[l.strip() for l in p.splitlines() if l.strip()] for p in pages]
    edges = Counter()
    for L in lines:
        edges.update({L[0], L[-1]} if L else set())
    cutoff = max(2, len(lines) // 2)
    repeated = {t for t, c in edges.items() if c >= cutoff}

    out = []
    for L in lines:
        n = len(L)
        # The eligible window is the top two and bottom two lines. On a 2- or
        # 3-line page those two windows overlap, so the indices are collected as
        # a set: the previous version walked a fixed (0, 1, -1, -2) and blanked
        # entries in place, so on a short page a later index read back a None the
        # loop had written and handed it to the regex. A page holding nothing but
        # a running header and a page number is exactly that shape, and a long
        # document always has one.
        #
        # Deciding which lines to drop before dropping any of them is what keeps
        # that from coming back: nothing is written into the line list at all, so
        # there is no sentinel to read back and the failure cannot be expressed.
        edge = {0, 1, n - 2, n - 1} & set(range(n))
        drop = {i for i in edge
                if L[i] in repeated or NUMBERISH.fullmatch(L[i])}
        out.append("\n".join(t for i, t in enumerate(L) if i not in drop))
    return out


def merge(pages: list[tuple[int, str]], markers: bool = True,
          certs: dict[int, float] | None = None) -> str:
    """Pages joined as continuous prose. With markers, each page block starts
    with an HTML comment carrying its physical page number (and certification
    when available) -- invisible in rendered markdown, parseable by RAG ingest
    as node metadata."""
    out = []
    for n, text in zip((p for p, _ in pages),
                       strip_noise([t for _, t in pages])):
        if not text.strip():
            continue
        if markers:
            tag = f"<!-- page {n}"
            if certs and n in certs:
                tag += f", cert {certs[n]:.2f}"
            out.append(tag + " -->\n\n" + text)
        else:
            out.append(text)
    return "\n\n".join(out) + "\n"


def learn_codec(pdf: Path, page_file, run_pool, args, total: int):
    """OCR the first 8 pages as a sample and learn the layer's corruption
    codec from their layer/OCR alignment. Returns (rules, seen, coverage) or
    None when the sample is too thin to trust. coverage is the fraction of the
    sample's OCR words the encoder can encode -- the codec's reach."""
    calib = [p for p in range(1, min(9, total + 1))]
    need = [p for p in calib if not page_file(p).exists()]
    if need:
        run_pool(need)
    refs = {p: page_file(p).read_text(encoding="utf-8") for p in calib}
    rules, seen = learn_rules([(text_layer(pdf, p), refs[p]) for p in calib])
    if len(rules) < 8:
        return None
    enc = build_encoder(rules, seen)
    vocab = set(w for ref in refs.values()
                for w in ref.replace("\f", "").split())
    coverage = sum(1 for w in vocab if w in enc) / len(vocab) if vocab else 0.0
    return rules, seen, coverage


def try_decode(pdf: Path, pages: list[int], page_file, scratch: Path, codec,
               args) -> int:
    """Calibrated layer decoding. Learn the layer's corruption rules from a
    small OCR'd sample, then decode every other page straight from the layer --
    no rasterization, no OCR -- as long as its layer is fully explained by the
    learned rules. Pages that fail the confidence gates are left for OCR.
    Returns the number of pages decoded."""
    rules, seen = codec
    n = 0
    for p in pages:
        if page_file(p).exists():
            continue
        layer = text_layer(pdf, p)
        chars = len("".join(layer.split()).replace("\f", ""))
        out, known, novel = decode_layer(layer, rules, seen)
        if chars and known / chars >= 0.98 and novel == 0:
            page_file(p).write_text(out, encoding="utf-8")
            (scratch / f"page-{p:04d}.md.decoded").touch()
            n += 1
    print(f"decode: {len(rules)} rules, {n} pages decoded from the layer",
          flush=True)
    return n


def plan_chunks(counts: list[tuple[Path, int]], shards: int, chunk: int
                ) -> list[list[tuple[Path, int, int]]]:
    """Cut every document into runs of `chunk` pages, then deal the runs into
    `shards` groups of roughly equal page count. Sharding by whole document
    leaves one runner with a 389-page document while others hold 17 pages; this
    levels them (measured 113-122 pages per group on this corpus).

    Groups are contiguous slices of the run list rather than round-robin, which
    keeps each group on a handful of documents. Every document a group touches
    costs one text-layer gate OCR, so interleaving would multiply that gate by
    the shard count for no balancing gain."""
    runs = [(p, s, min(s + chunk - 1, n))
            for p, n in counts for s in range(1, n + 1, chunk)]
    total = sum(n for _, n in counts)
    groups: list[list[tuple[Path, int, int]]] = [[] for _ in range(shards)]
    k = acc = 0
    for r in runs:
        # step to the next group once this one holds its share of the pages
        if k < shards - 1 and acc >= total * (k + 1) / shards:
            k += 1
        groups[k].append(r)
        acc += r[2] - r[1] + 1
    return groups


def spec_of(runs: list[tuple[Path, int, int]]) -> str:
    """The page runs of one document as a --pages spec: '1-10,101-110'."""
    return ",".join(f"{a}-{b}" for _, a, b in runs)


def paths_for(outdir: Path, name: str) -> tuple[Path, Path]:
    """A document's .md and its page cache. `name` is the path relative to --in
    without the suffix ('acts/12171_foo'), so the tree under document/ is
    mirrored under output/. That is what lets two subdirectories hold the same
    filename without colliding -- acts/ and acts-not-in-volume/ do exactly that
    for 13495 and 13534."""
    merged = outdir / f"{name}.md"
    return merged, merged.parent / f".{merged.stem}.pages"


def assemble(pdf: Path, outdir: Path, name: str, args) -> int:
    """Merge a document's page cache into its final .md. Split across shards, a
    document's pages arrive from several workers, so assembly happens once here
    -- after every cache has been overlaid -- rather than per worker. That is
    also what keeps strip_noise() honest: it infers running headers from what
    repeats across pages, which needs the whole document, not one worker's slice."""
    merged, scratch = paths_for(outdir, name)
    merged.parent.mkdir(parents=True, exist_ok=True)
    total = page_count(pdf)

    def page_file(pno: int) -> Path:
        return scratch / f"page-{pno:04d}.md"

    missing = [p for p in range(1, total + 1) if not page_file(p).exists()]
    if missing:
        # a gap means a worker died; writing anyway would produce a truncated .md
        # that looks complete to the skip check and to whatever ingests it
        raise RuntimeError(f"incomplete: {len(missing)} of {total} pages missing "
                           f"({runs_of(missing)})")

    certs = {}
    for p in range(1, total + 1):
        marker = scratch / f"page-{p:04d}.md.cert"
        if marker.exists():
            certs[p] = float(marker.read_text(encoding="utf-8"))
    have = [(p, page_file(p).read_text(encoding="utf-8")) for p in range(1, total + 1)]
    merged.write_text(merge(have, args.markers, certs or None), encoding="utf-8")
    shutil.rmtree(scratch)
    print(f"merged: {merged} ({total} pages)", flush=True)
    return total


def runs_of(pages: list[int]) -> str:
    """[1,2,3,9] -> '1-3, 9' -- gaps named as ranges, not a wall of numbers."""
    out, start = [], None
    for i, p in enumerate(pages):
        if start is None:
            start = p
        if i + 1 == len(pages) or pages[i + 1] != p + 1:
            out.append(f"{start}-{p}" if p != start else str(start))
            start = None
    return ", ".join(out)


def convert(pdf: Path, outdir: Path, name: str, args, pages_spec: str | None = None) -> int:
    """pages_spec overrides --pages for this document alone: a shard's share of a
    document is a set of disjoint page runs, different for every document it
    touches, so it cannot come from a single global flag."""
    spec = args.pages if pages_spec is None else pages_spec
    merged, scratch = paths_for(outdir, name)
    merged.parent.mkdir(parents=True, exist_ok=True)
    total = page_count(pdf)
    pages = parse_pages(spec, total)
    # An existing .md means this document is already converted; leave it alone so
    # a rerun only picks up what is new. Deliberately not an mtime comparison:
    # a fresh git checkout stamps every file at once, so mtime says nothing about
    # whether the .md is current. --force is how you redo one.
    # A finished run deletes its scratch dir, so a leftover one means the last run
    # was partial (crashed, or --pages) -- never skip in that case.
    if not spec and not args.force and merged.exists() and not scratch.exists():
        print(f"skip (already converted): {pdf}")
        return 0

    scratch.mkdir(parents=True, exist_ok=True)

    def page_file(pno: int) -> Path:
        return scratch / f"page-{pno:04d}.md"

    # Ground-truth gate: OCR page 1 and compare it against the text layer before
    # trusting the layer for the rest of the document. A legacy-derived layer can
    # be Devanagari yet ~40% wrong, so a ratio check alone is not enough -- the
    # match against what is actually printed decides. A scanned PDF has no layer,
    # so the gate (and its one OCR) is skipped entirely. The gate's OCR goes into
    # the page cache so the pool does not redo it -- but only when page 1 is ours
    # to publish: a shard that was not assigned page 1 would otherwise overwrite
    # the owning shard's (better, text-layer) copy when the caches are overlaid.
    probe = False
    if args.probe:
        layer = text_layer(pdf, 1)
        if layer.replace("\f", "").strip():
            if page_file(1).exists():
                ref = page_file(1).read_text(encoding="utf-8")
            else:
                with tempfile.TemporaryDirectory() as tmp:
                    ref = ocr(render(pdf, 1, Path(tmp), args.dpi), args.lang, args.psm)
                if 1 in pages:
                    page_file(1).write_text(ref, encoding="utf-8")
            agree = agreement(layer, ref)
            probe = agree >= LAYER_AGREEMENT
            print(f"text-layer gate: page-1 agreement {agree:.2f} "
                  f"({'trusting the layer' if probe else 'OCRing everything'})",
                  flush=True)

    lock = threading.Lock()
    done = 0
    probed = 0
    decoded = 0
    fast = 0
    escalated = 0
    certs: list[tuple[int, float]] = []

    def cert_marker(pno: int) -> Path:
        return scratch / f"page-{pno:04d}.md.cert"

    def one(pno: int) -> None:
        nonlocal done, probed, fast, escalated
        text = text_layer(pdf, pno) if probe else ""
        if usable(text):
            with lock:
                probed += 1
        else:
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                if args.fast:
                    # fast pass: low dpi + tesseract's own confidence decides
                    text, conf = ocr_fast(render(pdf, pno, tmp, args.fast_dpi),
                                          args.lang, args.psm)
                    if conf >= FAST_CONF:
                        with lock:
                            fast += 1
                    else:
                        text = ocr(render(pdf, pno, tmp, args.dpi),
                                   args.lang, args.psm)
                else:
                    text = ocr(render(pdf, pno, tmp, args.dpi), args.lang, args.psm)
        page_file(pno).write_text(text, encoding="utf-8")
        if enc is not None and not cert_marker(pno).exists():
            # certified OCR: re-encode through the learned codec and compare
            # with the page's own bytes; low-cert pages get one high-dpi redo
            cert, _ = certify_page(text, text_layer(pdf, pno), enc)
            if cert < CERT_MIN:
                with tempfile.TemporaryDirectory() as tmp:
                    text = ocr(render(pdf, pno, Path(tmp), args.cert_dpi),
                               args.lang, args.psm)
                page_file(pno).write_text(text, encoding="utf-8")
                cert, _ = certify_page(text, text_layer(pdf, pno), enc)
                with lock:
                    escalated += 1
            cert_marker(pno).write_text(f"{cert:.3f}", encoding="utf-8")
            with lock:
                certs.append((pno, cert))
        with lock:
            done += 1
            print(f"  page {pno}/{total} done ({done}/{len(pages)})", flush=True)

    def run_pool(pnos: list[int]) -> None:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for fut in [pool.submit(one, p) for p in pnos]:
                fut.result()

    # The layer's corruption codec, learned from an 8-page OCR sample, powers
    # both --decode (rewrite pages from the layer) and --certify (re-encode OCR
    # text back and compare with the page's own bytes). A thin sample disables
    # both: an unlearnable layer is safer left to plain OCR.
    codec = enc = None
    if (not probe and (args.decode or args.certify)
            and text_layer(pdf, 1).replace("\f", "").strip()):
        codec = learn_codec(pdf, page_file, run_pool, args, total)
        if codec is None:
            print("codec: 8 sample pages are too thin to learn from -- "
                  "decode/certify disabled", flush=True)
        else:
            rules, seen, coverage = codec
            if args.decode:
                decoded = try_decode(pdf, pages, page_file, scratch,
                                     (rules, seen), args)
            if args.certify:
                if coverage >= CODEC_COVERAGE:
                    enc = build_encoder(rules, seen)
                else:
                    print(f"codec: coverage {coverage:.0%} too thin to certify "
                          f"-- certification off", flush=True)

    def cached(pno: int) -> bool:
        # a decoded page is only reusable by another --decode run
        return (page_file(pno).exists()
                and (args.decode
                     or not (scratch / f"page-{pno:04d}.md.decoded").exists()))

    todo = [p for p in pages if not cached(p)]
    print(f"{pdf}: {len(pages)} pages ({len(todo)} to do, "
          f"{len(pages) - len(todo)} cached) -> {merged}", flush=True)

    t0 = time.time()
    run_pool(todo)

    cert_tail = ""
    if certs:
        mean = sum(c for _, c in certs) / len(certs)
        cert_tail = (f", cert mean {mean:.2f} over {len(certs)} pages "
                     f"({escalated} re-OCRed at {args.cert_dpi} dpi)")

    if spec:
        # Partial runs cache pages for later resume but never write a merged .md --
        # a partial <name>.md would look up to date to the skip check and silently
        # stay partial forever. Every shard lands here; --merge-only does the one
        # assembly pass once all the caches have been overlaid.
        print(f"pages {spec}: {len(pages)} pages cached in {scratch}, no merge "
              f"({probed} from text layer, {decoded} decoded, {fast} fast)"
              f"{cert_tail}", flush=True)
        return len(pages)

    cert_map = {n: c for n, c in certs} if certs else None
    have = [(p, page_file(p).read_text(encoding="utf-8")) for p in pages]
    merged.write_text(merge(have, args.markers, cert_map), encoding="utf-8")
    shutil.rmtree(scratch)  # only <name>.md survives a complete run
    print(f"done: {merged} ({len(have)} pages in {time.time() - t0:.0f}s, "
          f"{probed} from text layer, {decoded} decoded, {fast} fast)"
          f"{cert_tail}")
    return len(have)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="indir", default="documents")
    p.add_argument("--out", dest="outdir", default="output")
    p.add_argument("--pages", help="page range or sample, e.g. '169-173', "
                                   "'1-183/10' (every 10th) or '1-183:10' "
                                   "(exactly 10, spread evenly) (default: all)")
    # sched_getaffinity, not cpu_count: it respects an affinity mask or cgroup CPU
    # limit, so the pool matches the cores actually usable rather than the cores the
    # box happens to have. Each worker runs one tesseract at OMP_THREAD_LIMIT=1,
    # so this is a core count, not a thread count.
    p.add_argument("--workers", type=int, default=usable_cpus())
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--lang", default="nep+eng")
    # psm 6 (uniform block) beat psm 3 on the real corpus: column detection was
    # hoisting list numbers out of reading order. --psm 3 if a doc is truly multi-column.
    p.add_argument("--psm", default="6")
    p.add_argument("--probe", action=argparse.BooleanOptionalAction, default=True,
                   help="read pages with a usable text layer via pdftotext; "
                        "--no-probe forces OCR for everything")
    p.add_argument("--decode", action="store_true",
                   help="calibrate the text layer's corruption on an 8-page OCR "
                        "sample and decode the rest from the layer (lossy; "
                        "confidence-gated, but read the README before using)")
    p.add_argument("--markers", action=argparse.BooleanOptionalAction, default=True,
                   help="start each page with a <!-- page N --> comment, the "
                        "anchor RAG ingest parses into node metadata; "
                        "--no-markers gives pure prose")
    p.add_argument("--fast", action="store_true",
                   help="OCR at --fast-dpi first and keep it when tesseract's "
                        "mean confidence is high enough; only low-confidence "
                        "pages re-OCR at --dpi")
    p.add_argument("--fast-dpi", type=int, default=150)
    p.add_argument("--certify", action="store_true",
                   help="re-encode OCR'd pages through the learned layer codec "
                        "and compare with the PDF's own bytes; pages certified "
                        "below the threshold re-OCR once at --cert-dpi")
    p.add_argument("--cert-dpi", type=int, default=400)
    p.add_argument("--force", action="store_true",
                   help="reconvert documents even when their .md looks up to "
                        "date; use on CI, where a fresh checkout gives every "
                        "file the same mtime and the skip check is meaningless")
    p.add_argument("--shard", type=int, metavar="N",
                   help="OCR only this shard's share of the work (1-based); "
                        "pages are cached, and --merge-only assembles them")
    p.add_argument("--total-shards", type=int, metavar="N",
                   help="how many shards the work is split between")
    p.add_argument("--chunk", type=int, default=10, metavar="N",
                   help="pages per work unit when sharding (default 10); "
                        "smaller balances better but costs more per-document setup")
    p.add_argument("--merge-only", action="store_true",
                   help="assemble each document's cached pages into its .md and "
                        "stop; fails if any page is missing. Run this once after "
                        "every shard's cache has been overlaid into --out")
    p.add_argument("--only", metavar="REL",
                   help="convert just this one PDF, named relative to --in "
                        "('acts/12171_foo.pdf'). One job per document on CI")
    p.add_argument("--only-list", metavar="FILE",
                   help="convert the documents named in FILE, one path per line, "
                        "each relative to --in. A batch of whole documents: a "
                        "corpus of any size fits in a fixed number of CI jobs, "
                        "where one job per document would hit the 256-job matrix cap")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # sorted() is what makes sharding safe: every runner derives the same list
    # independently, so the slices below partition the corpus exactly once.
    pdfs = sorted(Path(args.indir).rglob("*.pdf"))
    if not pdfs:
        sys.exit(f"no PDFs found in {args.indir}/")

    if args.only and args.only_list:
        sys.exit("--only and --only-list are two ways to say the same thing")

    if args.only:
        want = Path(args.indir) / args.only
        if want not in pdfs:
            sys.exit(f"--only {args.only!r}: not a PDF under {args.indir}/")
        pdfs = [want]

    if args.only_list:
        wanted = [l.strip() for l in
                  Path(args.only_list).read_text(encoding="utf-8").splitlines()
                  if l.strip()]
        known = {str(p.relative_to(args.indir)): p for p in pdfs}
        missing = [w for w in wanted if w not in known]
        if missing:
            sys.exit(f"--only-list: {len(missing)} of {len(wanted)} not found under "
                     f"{args.indir}/, first is {missing[0]!r}")
        pdfs = [known[w] for w in wanted]
        print(f"batch of {len(pdfs)} documents", flush=True)

    # the relative path, minus .pdf -- output/ mirrors the tree under --in
    def flat(pdf: Path) -> str:
        return str(pdf.relative_to(args.indir).with_suffix(""))

    # work[pdf] = the page spec for this run, or None for the whole document
    work: list[tuple[Path, str | None]] = [(p, None) for p in pdfs]

    if args.merge_only and (args.shard is not None or args.total_shards is not None):
        sys.exit("--merge-only assembles every document; it takes no shard")

    if (args.only or args.only_list) and (args.shard is not None
                                          or args.total_shards is not None):
        sys.exit("--only/--only-list already name this job's work; it takes no shard")

    if args.shard is not None or args.total_shards is not None:
        if args.shard is None or args.total_shards is None:
            sys.exit("--shard and --total-shards must be given together")
        if not 1 <= args.shard <= args.total_shards:
            sys.exit(f"--shard must be 1..{args.total_shards}, got {args.shard}")
        if args.chunk < 1:
            sys.exit(f"--chunk must be at least 1, got {args.chunk}")
        counts = [(p, page_count(p)) for p in pdfs]
        mine = plan_chunks(counts, args.total_shards, args.chunk)[args.shard - 1]
        work = [(pdf, spec_of([r for r in mine if r[0] == pdf]))
                for pdf in dict.fromkeys(r[0] for r in mine)]
        npages = sum(b - a + 1 for _, a, b in mine)
        print(f"shard {args.shard}/{args.total_shards}: {npages} pages across "
              f"{len(work)} documents", flush=True)

    failed = 0
    for pdf, spec in work:
        try:
            if args.merge_only:
                assemble(pdf, outdir, flat(pdf), args)
            else:
                convert(pdf, outdir, flat(pdf), args, pages_spec=spec)
        except Exception as e:
            failed += 1
            err = getattr(e, "stderr", "") or e
            print(f"FAILED: {pdf}: {err}", file=sys.stderr)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
