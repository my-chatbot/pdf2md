#!/usr/bin/env python3
"""Nepali PDF -> Markdown via Tesseract OCR, shaped for a RAG pipeline.

Pages with a genuine Unicode text layer are read straight from the PDF with
pdftotext; everything else (scanned, or legacy-font Preeti/Kantipur, whose text
layer is ASCII bytes mapped through a non-Unicode font) is rasterized and OCR'd.
Running headers, footers and page numbers are stripped so they don't end up in
every retrieved chunk. One .md per document, nothing else.

With --shard and --total-shards, the script processes only a subset of pages
and writes a partial Markdown file.
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
    ('5/10', '5:10') run to the last page; None -> all. 1-based."""
    if not s:
        return list(range(1, total + 1))
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
        keep = list(L)
        for idx in (0, 1, -1, -2):
            if not keep or len(keep) <= abs(idx):
                continue
            t = keep[idx]
            if t in repeated or NUMBERISH.fullmatch(t):
                keep[idx] = None
        out.append("\n".join(t for t in keep if t is not None))
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


def convert(pdf: Path, outdir: Path, name: str, args) -> int:
    merged = outdir / f"{name}.md"
    scratch = outdir / f".{name}.pages"
    total = page_count(pdf)
    pages = parse_pages(args.pages, total)
    # A finished run deletes its scratch dir, so a leftover one means the last run was
    # partial (crashed, or --pages) -- never skip in that case.
    if (not args.pages and merged.exists() and not scratch.exists()
            and merged.stat().st_mtime >= pdf.stat().st_mtime):
        print(f"skip (up to date): {pdf}")
        return 0

    scratch.mkdir(parents=True, exist_ok=True)

    def page_file(pno: int) -> Path:
        return scratch / f"page-{pno:04d}.md"

    # Ground-truth gate: OCR page 1 and compare it against the text layer before
    # trusting the layer for the rest of the document. A legacy-derived layer can
    # be Devanagari yet ~40% wrong, so a ratio check alone is not enough -- the
    # match against what is actually printed decides. A scanned PDF has no layer,
    # so the gate (and its one OCR) is skipped entirely. The gate's OCR goes into
    # the page cache so the pool does not redo it.
    probe = False
    if args.probe:
        layer = text_layer(pdf, 1)
        if layer.replace("\f", "").strip():
            if page_file(1).exists():
                ref = page_file(1).read_text(encoding="utf-8")
            else:
                with tempfile.TemporaryDirectory() as tmp:
                    ref = ocr(render(pdf, 1, Path(tmp), args.dpi), args.lang, args.psm)
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

    # ------------------------------------------------------------------
    # SHARDING LOGIC – write partial file if --shard is given
    # ------------------------------------------------------------------
    have = [(p, page_file(p).read_text(encoding="utf-8")) for p in pages]
    cert_map = {n: c for n, c in certs} if certs else None

    if args.shard and args.total_shards:
        # Write a partial Markdown file for this shard
        partial_file = outdir / f"{name}_shard_{args.shard}.md"
        partial_text = merge(have, args.markers, cert_map)
        partial_file.write_text(partial_text, encoding="utf-8")
        print(f"Partial shard {args.shard} saved to {partial_file}")
        # Keep scratch directory for potential resumption
        return len(have)

    # Normal (non‑sharded) mode: write the final merged file and clean up
    merged.write_text(merge(have, args.markers, cert_map), encoding="utf-8")
    shutil.rmtree(scratch)
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
    p.add_argument("--workers", type=int, default=os.cpu_count())
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
    # --- NEW: sharding arguments ---
    p.add_argument("--shard", type=int, help="shard number (1‑based)")
    p.add_argument("--total-shards", type=int, help="total number of shards")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(Path(args.indir).rglob("*.pdf"))
    if not pdfs:
        sys.exit(f"no PDFs found in {args.indir}/")

    failed = 0
    for pdf in pdfs:
        rel = pdf.relative_to(args.indir).with_suffix("")
        try:
            convert(pdf, outdir, "__".join(rel.parts), args)
        except Exception as e:
            failed += 1
            err = getattr(e, "stderr", "") or e
            print(f"FAILED: {pdf}: {err}", file=sys.stderr)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
