"""
Evaluate Strategy A (no-leak) captions.

QA metrics:
1. Format compliance: starts with "A dermoscopic image of", ends with period.
2. Length compliance: 15-25 words.
3. NO-LEAK verification: caption must NOT contain forbidden tokens
   (diagnosis words, anatomic sites, demographics). This is the KEY metric
   for Strategy A — any leak invalidates the no-leak guarantee.
4. Lexical diversity: unique trigrams / total trigrams.
5. Class-distinguishability sanity check: do captions for melanoma images
   actually use more "irregularity"-related vocabulary than nevus captions?
   (We're NOT supervising this — we want to verify GPT-4V isn't producing
   identical generic captions across all classes, which would mean text has
   zero signal for downstream segmentation.)

Usage:
    python caption_gen/evaluate.py --captions outputs/captions/captions.jsonl
    open outputs/captions/review.html
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Allow `python caption_gen/evaluate.py` without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from caption_gen.prompts import FORBIDDEN_TOKENS


FORMAT_RE = re.compile(r"^A dermoscopic image of .+\.$", re.IGNORECASE)


# Vocabulary categories used for class-distinguishability sanity check.
IRREGULARITY_VOCAB = {
    "asymmetric", "asymmetry", "irregular", "irregularity", "notched",
    "jagged", "blurred", "ill-defined", "uneven",
}
MULTICOLOR_VOCAB = {
    "multi-colored", "multicolored", "multiple colors", "varied",
    "heterogeneous", "blue-whitish", "veil",
}
SK_VOCAB = {
    "milia", "milia-like", "comedo", "comedo-like", "cerebriform",
    "fissures", "ridges", "moth-eaten",
}


def word_count(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s))


def trigrams(words: list[str]) -> list[tuple]:
    return [tuple(words[i:i + 3]) for i in range(len(words) - 2)]


def find_forbidden(caption: str) -> list[str]:
    """Return list of forbidden tokens found in caption."""
    cap_lower = caption.lower()
    found: list[str] = []
    for tok in FORBIDDEN_TOKENS:
        # Word-boundary match for single words; substring for multi-word tokens.
        if " " in tok or "-" in tok:
            if tok in cap_lower:
                found.append(tok)
        else:
            if re.search(rf"\b{re.escape(tok)}\b", cap_lower):
                found.append(tok)
    return found


def vocab_hits(caption: str, vocab: set[str]) -> int:
    cap_lower = caption.lower()
    return sum(1 for v in vocab if v in cap_lower)


def evaluate(records: list[dict]) -> dict:
    n = len(records)
    n_format_ok = 0
    n_length_ok = 0
    n_no_leak = 0
    n_errors = 0
    word_counts: list[int] = []
    all_trigrams: list[tuple] = []
    leak_examples: list[dict] = []

    cls_vocab: dict[str, dict[str, int]] = defaultdict(
        lambda: {"irregular": 0, "multicolor": 0, "sk": 0, "n": 0}
    )

    for r in records:
        if r.get("error") or not r.get("caption"):
            n_errors += 1
            continue
        cap = r["caption"].strip()
        wc = word_count(cap)
        word_counts.append(wc)

        if FORMAT_RE.match(cap):
            n_format_ok += 1
        if 15 <= wc <= 25:
            n_length_ok += 1

        leaks = find_forbidden(cap)
        if not leaks:
            n_no_leak += 1
        else:
            leak_examples.append({
                "image_id": r["image_id"],
                "leaked_tokens": leaks,
                "caption": cap,
            })

        all_trigrams.extend(trigrams(re.findall(r"\b\w+\b", cap.lower())))

        cls = r.get("diagnosis_gt", "unknown")
        bucket = cls_vocab[cls]
        bucket["n"] += 1
        bucket["irregular"] += vocab_hits(cap, IRREGULARITY_VOCAB)
        bucket["multicolor"] += vocab_hits(cap, MULTICOLOR_VOCAB)
        bucket["sk"] += vocab_hits(cap, SK_VOCAB)

    tri_counter = Counter(all_trigrams)
    diversity = len(tri_counter) / max(sum(tri_counter.values()), 1)

    n_valid = n - n_errors

    cls_rates: dict[str, dict[str, float]] = {}
    for cls, b in cls_vocab.items():
        n_cls = max(b["n"], 1)
        cls_rates[cls] = {
            "n": b["n"],
            "irregular_per_caption": round(b["irregular"] / n_cls, 2),
            "multicolor_per_caption": round(b["multicolor"] / n_cls, 2),
            "sk_per_caption": round(b["sk"] / n_cls, 2),
        }

    return {
        "n_total": n,
        "n_errors": n_errors,
        "n_valid": n_valid,
        "format_compliance_pct": round(100 * n_format_ok / max(n_valid, 1), 1),
        "length_compliance_pct": round(100 * n_length_ok / max(n_valid, 1), 1),
        "no_leak_compliance_pct": round(100 * n_no_leak / max(n_valid, 1), 1),
        "trigram_diversity": round(diversity, 3),
        "word_count_min": min(word_counts) if word_counts else None,
        "word_count_max": max(word_counts) if word_counts else None,
        "word_count_avg": round(sum(word_counts) / max(len(word_counts), 1), 1),
        "leak_examples": leak_examples[:10],
        "class_vocab_usage": cls_rates,
        "class_distinguishability": _distinguishability_verdict(cls_rates),
        "verdict": _verdict(n_format_ok, n_no_leak, n_valid, diversity),
    }


def _distinguishability_verdict(cls_rates: dict) -> str:
    mel = cls_rates.get("melanoma", {})
    nev = cls_rates.get("nevus", {})
    sk = cls_rates.get("seborrheic keratosis", {})
    notes = []

    if mel.get("irregular_per_caption", 0) > nev.get("irregular_per_caption", 0):
        notes.append("OK: melanoma > nevus on irregularity vocab")
    else:
        notes.append("WARN: melanoma not more irregular-described than nevus")

    if mel.get("multicolor_per_caption", 0) > nev.get("multicolor_per_caption", 0):
        notes.append("OK: melanoma > nevus on multicolor vocab")
    else:
        notes.append("WARN: melanoma not more multicolor-described than nevus")

    if sk.get("sk_per_caption", 0) > nev.get("sk_per_caption", 0):
        notes.append("OK: SK > nevus on SK-specific vocab")
    else:
        notes.append("WARN: SK not using SK-specific vocab more than nevus")

    return "; ".join(notes)


def _verdict(n_format_ok: int, n_no_leak: int, n_valid: int, diversity: float) -> str:
    if n_valid == 0:
        return "NO_DATA"
    fmt_pct = 100 * n_format_ok / n_valid
    leak_pct = 100 * n_no_leak / n_valid

    if leak_pct < 100:
        return (f"FAIL — {n_valid - n_no_leak} caption(s) leaked GT vocab. "
                "Tighten FORBIDDEN list or strengthen system prompt.")
    if fmt_pct >= 95 and diversity >= 0.4:
        return "GOOD — proceed to scale (no leakage detected)."
    if fmt_pct >= 80:
        return "OK — iterate prompt before scaling."
    return "POOR — major prompt revision needed."


def render_html(records: list[dict], qa: dict) -> str:
    """Self-contained HTML viewer."""
    rows_html = []
    for r in records:
        cap = r.get("caption", "")
        format_ok = bool(FORMAT_RE.match(cap.strip())) if cap else False
        wc = word_count(cap) if cap else 0
        length_ok = 15 <= wc <= 25
        leaks = find_forbidden(cap) if cap else []
        no_leak = len(leaks) == 0
        err = r.get("error")

        b_fmt = ('<span class="ok">FORMAT_OK</span>' if format_ok
                 else '<span class="bad">FORMAT_BAD</span>')
        b_len = (f'<span class="ok">LEN_OK ({wc})</span>' if length_ok
                 else f'<span class="bad">LEN_BAD ({wc})</span>')
        b_leak = ('<span class="ok">NO_LEAK</span>' if no_leak
                  else f'<span class="bad">LEAK: {", ".join(leaks)}</span>')
        err_html = f'<div class="err">{err}</div>' if err else ""

        img_html = ""
        if r.get("image_path"):
            img_html = f'<img src="file://{r["image_path"]}" />'
        if r.get("mask_path"):
            img_html += f'<img src="file://{r["mask_path"]}" class="mask" />'

        rows_html.append(f"""
        <div class="card">
          <div class="img-wrap">
            {img_html}
          </div>
          <div class="meta">
            <div><b>{r['image_id']}</b>
              &middot; <span class="gtdx">gt_dx={r.get('diagnosis_gt', '?')}</span>
              <span class="gt-note">(stored for QA only — was NOT shown to GPT-4V)</span></div>
            <div class="caption">{cap or '<i>(no caption)</i>'}</div>
            <div class="badges">{b_fmt} {b_len} {b_leak}
              <span class="cost">${r.get('cost_usd', 0):.4f}</span>
              <span class="cost">{r.get('latency_s', 0):.1f}s</span>
            </div>
            {err_html}
          </div>
        </div>
        """)

    qa_simple = {k: v for k, v in qa.items()
                 if k not in {"leak_examples", "class_vocab_usage"}}
    qa_html = "<table>" + "".join(
        f"<tr><td>{k}</td><td><b>{v}</b></td></tr>"
        for k, v in qa_simple.items()
    ) + "</table>"

    cls_html = "<table><tr><th>class</th><th>n</th>" \
               "<th>irregular/cap</th><th>multicolor/cap</th>" \
               "<th>SK-vocab/cap</th></tr>"
    for cls, b in qa.get("class_vocab_usage", {}).items():
        cls_html += f"<tr><td>{cls}</td><td>{b['n']}</td>" \
                    f"<td>{b['irregular_per_caption']}</td>" \
                    f"<td>{b['multicolor_per_caption']}</td>" \
                    f"<td>{b['sk_per_caption']}</td></tr>"
    cls_html += "</table>"

    leak_html = ""
    if qa.get("leak_examples"):
        leak_html = "<h3>Leak examples (first 10)</h3><ul>"
        for ex in qa["leak_examples"]:
            leak_html += (f"<li><b>{ex['image_id']}</b> "
                          f"<span class='bad'>{', '.join(ex['leaked_tokens'])}</span>: "
                          f"{ex['caption']}</li>")
        leak_html += "</ul>"

    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>ISIC 2017 captions — Strategy A review</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 1100px; margin: 20px auto; padding: 0 10px; }}
h1, h2 {{ font-weight: 600; }}
.banner {{ background: #fff8c4; border-left: 4px solid #c4a000; padding: 10px 14px; margin-bottom: 16px; border-radius: 4px; }}
.card {{ border: 1px solid #ddd; border-radius: 8px; margin: 12px 0; padding: 12px;
  display: grid; grid-template-columns: 320px 1fr; gap: 16px; }}
.img-wrap {{ display: flex; flex-direction: column; gap: 4px; }}
.img-wrap img {{ width: 320px; height: auto; border: 1px solid #eee; border-radius: 4px; }}
.caption {{ font-size: 15px; margin: 6px 0; }}
.badges {{ margin-top: 6px; }}
.ok {{ background: #d4f4dd; color: #114; padding: 2px 6px; border-radius: 4px; margin-right: 4px; font-size: 12px; }}
.bad {{ background: #ffd0d0; color: #800; padding: 2px 6px; border-radius: 4px; margin-right: 4px; font-size: 12px; }}
.gtdx {{ color: #555; font-family: monospace; }}
.gt-note {{ color: #888; font-size: 11px; font-style: italic; }}
.cost {{ color: #888; font-size: 12px; margin-left: 8px; }}
.err {{ color: #800; font-size: 13px; margin-top: 4px; }}
table {{ border-collapse: collapse; margin: 8px 0; }}
table td, table th {{ padding: 4px 12px; border-bottom: 1px solid #eee; text-align: left; }}
table th {{ background: #f5f5f5; }}
code {{ background: #f0f0f0; padding: 1px 4px; border-radius: 3px; }}
</style></head><body>
<div class="banner">
  <b>Methodology: Strategy A (strict no-leak)</b><br>
  GPT-4V received ONLY the image — no diagnosis, no Part 2 features, no
  age/sex/anatomic-site. The <code>gt_dx</code> shown next to each caption
  is from ground truth and is used here for QA inspection only.
</div>

<h1>ISIC 2017 captions — review</h1>

<h2>QA summary</h2>
{qa_html}

<h2>Class-distinguishability check</h2>
<p style="color:#555">If GPT-4V is "seeing" class differences from the image alone,
melanoma captions should use more irregularity/multicolor vocab than nevus,
and seborrheic-keratosis captions should use more SK-specific terms.</p>
{cls_html}

{leak_html}

<h2>Captions ({len(records)})</h2>
{''.join(rows_html)}
</body></html>"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--captions", required=True,
                   help="Path to captions JSONL file.")
    p.add_argument("--out-dir", default=None,
                   help="Directory for review.html and qa.json output. "
                        "Defaults to the same directory as --captions.")
    return p.parse_args()


def main():
    args = parse_args()
    jsonl_path = Path(args.captions)
    if not jsonl_path.exists():
        print(f"[error] {jsonl_path} not found. Run caption_gen/generate.py first.")
        return

    out_dir = Path(args.out_dir) if args.out_dir else jsonl_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    review_html = out_dir / "review.html"
    report_json = out_dir / "qa.json"

    records = [json.loads(l) for l in open(jsonl_path, encoding="utf-8") if l.strip()]
    qa = evaluate(records)
    report_json.write_text(json.dumps(qa, indent=2))
    review_html.write_text(render_html(records, qa), encoding="utf-8")
    print(json.dumps(qa, indent=2))
    print(f"\n[done] HTML viewer: {review_html}")
    print(f"       JSON report:  {report_json}")


if __name__ == "__main__":
    main()
