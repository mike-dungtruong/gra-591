# Caption Generation Strategy — Strategy A (Strict No-Leak)

## What it is

Strategy A generates dermoscopy captions using GPT-4o in a **zero-shot image-only** setting:
the model receives the raw dermoscopic image and nothing else — no patient metadata, no ground-truth
diagnosis, no Part 2 superpixel features, no anatomic-site label.

The resulting `(image, caption) → mask` supervision is honest: the text branch cannot shortcut
to the label because the caption itself was generated without access to any label.

## Why this matters

This project uses captions as soft text guidance at each decoder stage (via TGCM). If GPT-4o were
shown the diagnosis while writing captions, the text branch would trivially learn "melanoma → gate
X" and gain an unfair advantage — one that would evaporate at inference when no label is available.
Strategy A prevents this by construction.

## Prompt design

The system prompt (`caption_gen/prompts.py`) enforces:

1. **Format**: exactly one sentence, 15–25 words, starting with "A dermoscopic image of", ending
   with a period.
2. **Observation-only vocabulary**: shape, border quality, colors, dermoscopic structures
   (pigment network, globules, streaks, blue-whitish veil, milia-like cysts, etc.).
3. **Forbidden vocabulary**: any diagnosis word (melanoma, nevus, seborrheic, benign, malignant,
   cancer, atypical, dysplastic, …), any anatomic site (back, arm, scalp, …), any demographic
   (age, sex). See `FORBIDDEN_TOKENS` in `prompts.py` for the full list.
4. **Conservative**: "if uncertain, omit" — a shorter accurate caption beats a longer hallucinated
   one.
5. **Low temperature** (0.1): reduces variance and hallucination at the cost of minor lexical
   uniformity.

Example output:
> A dermoscopic image of an asymmetric multi-colored lesion with irregular notched borders and a
> visible blue-whitish veil.

## Post-hoc QA (`caption_gen/evaluate.py`)

Before scaling, run QA on the pilot set:

```bash
python caption_gen/evaluate.py --captions outputs/captions/captions_train.jsonl
```

The QA gate checks:
- **No-leak compliance** (must be 100% before scaling) — any caption containing a forbidden token
  fails this check. A single failure invalidates the no-leak guarantee.
- Format compliance (≥95% acceptable).
- Length compliance (15–25 words).
- Trigram diversity (≥0.4 means captions are not repetitively identical).
- Class-distinguishability sanity (melanoma captions should use more irregularity/multicolor
  vocabulary than nevus; SK captions should use SK-specific terms more).

## Prior work contrast

| Work | Captioner sees ground truth? | Our case |
|---|---|---|
| **SkinCAP** (dermatologist captions) | Yes — dermatologist wrote captions knowing the diagnosis | Different use case: VLM pretraining, not closed-loop seg training |
| **BiomedParse** | GPT-4 for label harmonization with ontology | Not ungrounded description; ontology-guided |
| **LViT** | Structured radiology report (clinician-written, diagnosis-aware) | Report-based seg — clinically justified but label-aware |
| **This work** | No — image only | Strategy A; maximally honest for text-guided seg |

## Ablation note

An alternative **Strategy B** (anchored) would provide the diagnosis label to GPT-4o and ask it
to write a caption that describes visual ABCD criteria consistent with that diagnosis. This would
produce more clinically accurate captions but would introduce label leakage. Strategy B is a
reasonable ablation to evaluate whether caption quality (vs. no-leak honesty) is the bottleneck.

## Methods paragraph (draft)

> We generate per-image dermoscopy captions using GPT-4o in a zero-shot image-only setting
> (Strategy A). The model receives only the raw dermoscopic image; no ground-truth diagnosis,
> patient metadata, or anatomic site is provided. Each caption is constrained to a single sentence
> of 15–25 words describing observable visual attributes: lesion shape, border regularity, color
> distribution, and dermoscopic structures (pigment network, globules, streaks, blue-whitish veil,
> milia-like cysts). A forbidden-vocabulary filter post-hoc verifies that no diagnosis or
> demographic terms appear in any generated caption. This design ensures that the
> (image, text) → mask supervision is honest: the text branch cannot exploit a label shortcut,
> and any Dice/IoU improvement over the no-text baseline can be attributed to the TGCM's capacity
> to use visually grounded textual guidance.
