"""
Caption generation: Strategy A (NO GROUND TRUTH LEAKAGE).

GPT-4V receives ONLY the dermoscopy image — no metadata, no diagnosis,
no Part 2 features. This makes (image, caption) -> mask supervision
honest and prevents text-shortcut shortcuts in downstream training.

Metadata.csv is still loaded — but ONLY used for stratified sampling
(we want a balanced 15+15+20 across the three classes), NOT for prompt
content. The diagnosis label is also stored in the output JSONL solely
as ground truth for later QA evaluation (e.g., does GPT-4V's caption
of melanoma images actually describe more melanoma-like features than
nevus images?).

Usage:
    export OPENAI_API_KEY='sk-...'
    python caption_gen/generate.py --config configs/caption_gen.yaml --split train --n 1500 --resume
    python caption_gen/generate.py --config configs/caption_gen.yaml --split val   --n 650  --resume

Output:
    outputs/captions/captions_<split>.jsonl
    outputs/captions/caption_summary_<split>.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

# Allow `python caption_gen/generate.py` without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from caption_gen.client import GPT4oClient
from caption_gen.data_loader import (
    build_samples, load_metadata, stratified_sample, class_distribution
)
from caption_gen.prompts import build_messages


def load_config(config_path: str | None) -> dict:
    if config_path is None:
        return {}
    try:
        import yaml
    except ImportError:
        raise ImportError("pyyaml is required to use --config. Run: pip install pyyaml")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None,
                   help="Path to caption_gen.yaml. CLI flags override YAML values.")
    p.add_argument("--n", type=int, default=None,
                   help="Total images. Default split: 30%% mel + 30%% nevus + "
                        "40%% sk (matches ISIC 2017 task 3 evaluation balance).")
    p.add_argument("--model", default=None,
                   choices=["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"])
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--resume", action="store_true",
                   help="Skip image_ids already in the output JSONL.")
    p.add_argument("--no-stratify", action="store_true",
                   help="Skip stratified sampling (works without metadata.csv). "
                        "Will random-sample N images.")
    p.add_argument("--temperature", type=float, default=None,
                   help="Lower = more deterministic. Strategy A uses 0.1 to "
                        "minimize variance / hallucination.")
    p.add_argument("--out", default=None,
                   help="Output JSONL path. Defaults to output.dir/captions_<split>.jsonl "
                        "from config, or ./outputs/captions/captions_<split>.jsonl.")
    p.add_argument("--summary", default=None,
                   help="Summary JSON path.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print sample selection + prompt only, no API calls.")
    return p.parse_args()


def resolve(args, cfg: dict):
    """Merge YAML config with CLI overrides. CLI always wins."""
    api = cfg.get("api", {})
    sampling = cfg.get("sampling", {})
    output = cfg.get("output", {})

    out_dir = Path(output.get("dir", "./outputs/captions"))
    split = args.split

    model = args.model or api.get("model", "gpt-4o")
    temperature = args.temperature if args.temperature is not None else api.get("temperature", 0.1)
    seed = args.seed if args.seed is not None else sampling.get("seed", 42)

    # n defaults depend on split if not set
    if args.n is not None:
        n = args.n
    else:
        n = 1500 if split == "train" else 650

    out_file = args.out or str(out_dir / f"captions_{split}.jsonl")
    summary_file = args.summary or str(out_dir / f"caption_summary_{split}.json")

    return {
        "model": model,
        "temperature": temperature,
        "max_tokens": api.get("max_tokens", 80),
        "timeout": api.get("timeout", 60.0),
        "max_retries": api.get("max_retries", 6),
        "seed": seed,
        "n": n,
        "out_file": out_file,
        "summary_file": summary_file,
    }


def scan_output(out_path: Path) -> tuple[set[str], set[str]]:
    """Scan the output JSONL and return (succeeded_ids, errored_ids).

    succeeded_ids: have a non-empty caption and no error — skipped on --resume.
    errored_ids:   have an error or empty caption — retried on --resume.
    """
    if not out_path.exists():
        return set(), set()
    succeeded: set[str] = set()
    errored: set[str] = set()
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                iid = rec.get("image_id")
                if not iid:
                    continue
                if rec.get("caption") and not rec.get("error"):
                    succeeded.add(iid)
                else:
                    errored.add(iid)
            except (json.JSONDecodeError, KeyError):
                pass
    # An image that later succeeded may still have an earlier error line;
    # don't count it as errored.
    errored -= succeeded
    return succeeded, errored


def main():
    args = parse_args()
    cfg = load_config(args.config)
    opts = resolve(args, cfg)

    out_path = Path(opts["out_file"])
    summary_path = Path(opts["summary_file"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[info] METHODOLOGY: Strategy A (strict no-leak). "
          f"GPT-4V sees ONLY the image.")
    print(f"[info] loading metadata + samples (split={args.split})...")
    metadata = load_metadata()

    # Dry-run only needs to show the prompt structure, not call APIs.
    # Auto-fall-back to no-stratify so the user can verify prompt without
    # having to download metadata first.
    if args.dry_run and not metadata and not args.no_stratify:
        print("[info] dry-run: no metadata found, auto-enabling --no-stratify")
        args.no_stratify = True

    if not metadata and not args.no_stratify:
        print("[warn] No metadata.csv found AND --no-stratify not set.")
        print("       Strategy A doesn't need metadata for the prompt, but")
        print("       we still need diagnosis labels for stratified sampling.")
        print("       Re-run with --no-stratify for random sampling (no metadata needed).")
        sys.exit(1)

    samples = build_samples(split=args.split, metadata=metadata)
    print(f"[info] discovered {len(samples)} samples with masks")
    if metadata:
        print(f"[info] full distribution: {class_distribution(samples)}")

    n = opts["n"]
    seed = opts["seed"]

    if args.no_stratify:
        rng = random.Random(seed)
        picked = rng.sample(samples, min(n, len(samples)))
        print(f"[info] random-sampled {len(picked)} (no stratification)")
    else:
        # Stratified sampling — balanced for evaluation purposes.
        samples = [s for s in samples if s.diagnosis in
                   {"melanoma", "nevus", "seborrheic keratosis"}]
        n_per_class = {
            "melanoma": int(n * 0.3),
            "nevus": int(n * 0.3),
            "seborrheic keratosis": n - 2 * int(n * 0.3),
        }
        print(f"[info] target per-class: {n_per_class}")
        picked = stratified_sample(samples, n_per_class, seed=seed)
        print(f"[info] stratified-sampled {len(picked)}")

    if args.resume:
        done, errored = scan_output(out_path)
        n_retry = len(errored & {s.image_id for s in picked})
        print(f"[info] resume: {len(done)} succeeded (skipping), "
              f"{n_retry} errored (will retry)")
        picked = [s for s in picked if s.image_id not in done]

    if args.dry_run:
        msgs = build_messages(image_b64_data_url="data:image/jpeg;base64,<IMAGE_OMITTED>")
        print("\n[dry-run] sample messages (Strategy A — no metadata):")
        print("=" * 60)
        print("SYSTEM:")
        print(msgs[0]["content"][:1200] + "...")
        print()
        print("USER:")
        print(msgs[1]["content"][0]["text"])
        print(f"  [+ image of size N bytes]")
        print("=" * 60)
        print(f"\n[dry-run] would call API for {len(picked)} images")
        print(f"[dry-run] sample image_ids: {[s.image_id for s in picked[:5]]}")
        return

    client = GPT4oClient(
        model=opts["model"],
        timeout=opts["timeout"],
        max_retries=opts["max_retries"],
    )

    total_cost = 0.0
    total_latency = 0.0
    n_ok = 0
    n_err = 0
    t_start = time.time()

    with open(out_path, "a", encoding="utf-8") as fout:
        for i, s in enumerate(picked, 1):
            print(f"[{i}/{len(picked)}] {s.image_id} (gt_dx={s.diagnosis}) ...",
                  end=" ", flush=True)
            try:
                # NOTE: we pass ONLY the image to the prompt builder.
                # diagnosis/age/sex/site are stored in record but NOT in prompt.
                try:
                    msgs = build_messages(image_b64_data_url=s.image_data_url())
                except OSError as load_err:
                    n_err += 1
                    print(f"ERR (image load: {load_err})")
                    fout.write(json.dumps({
                        "image_id": s.image_id, "split": args.split,
                        "diagnosis_gt": s.diagnosis, "caption": "",
                        "error": f"image_load: {load_err}",
                    }, ensure_ascii=False) + "\n")
                    fout.flush()
                    continue
                res = client.caption(s.image_id, msgs,
                                     max_tokens=opts["max_tokens"],
                                     temperature=opts["temperature"])
                if res.error:
                    n_err += 1
                    print(f"ERR ({res.error[:80]})")
                else:
                    n_ok += 1
                    total_cost += res.cost_usd
                    total_latency += res.latency_s
                    print(f"ok ${res.cost_usd:.4f} {res.latency_s:.1f}s")
                    print(f"     -> {res.caption[:120]}")

                # Diagnosis stored ONLY for downstream QA — was NOT in prompt.
                rec = {
                    "image_id": s.image_id,
                    "split": args.split,
                    "diagnosis_gt": s.diagnosis,        # for QA only
                    "age": s.age,                       # for QA only
                    "sex": s.sex,                       # for QA only
                    "anatomic_site": s.anatomic_site,   # for QA only
                    "image_path": str(s.image_path),
                    "mask_path": str(s.mask_path),
                    "model": res.model,
                    "caption": res.caption,
                    "prompt_tokens": res.prompt_tokens,
                    "completion_tokens": res.completion_tokens,
                    "cost_usd": res.cost_usd,
                    "latency_s": res.latency_s,
                    "temperature": opts["temperature"],
                    "strategy": "A_no_leak",
                    "error": res.error,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
            except KeyboardInterrupt:
                print("\n[info] interrupted, partial output saved.")
                break

    elapsed = time.time() - t_start
    summary = {
        "strategy": "A_no_leak",
        "model": opts["model"],
        "temperature": opts["temperature"],
        "n_requested": len(picked),
        "n_ok": n_ok,
        "n_err": n_err,
        "total_cost_usd": round(total_cost, 4),
        "avg_latency_s": round(total_latency / max(n_ok, 1), 2),
        "wall_time_s": round(elapsed, 1),
        "extrapolated_cost_2000_imgs": round(total_cost / max(n_ok, 1) * 2000, 2),
        "out_file": str(out_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] {summary}")


if __name__ == "__main__":
    main()
