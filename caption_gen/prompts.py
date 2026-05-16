"""
Prompt templates for ISIC 2017 caption generation — Strategy A (strict no-leak).

METHODOLOGY: NO GROUND TRUTH IS PROVIDED TO THE VLM
=====================================================
We adopt Strategy A from the methodology discussion:
- GPT-4V receives ONLY the dermoscopy image, no metadata, no diagnosis,
  no Part 2 features, no age/sex/site, no mask reference.
- Caption describes only what is visually observable from the image.
- This eliminates label leakage in downstream text-guided segmentation
  training: the (image, text) -> mask supervision is honest.

Trade-off: GPT-4V cannot anchor on diagnosis, so caption quality is fully
dependent on its zero-shot dermoscopy reasoning. Expect ~10-20% captions
to need manual review or filtering.

Mitigations baked into the prompt:
- Conservative prompting ("OMIT if uncertain")
- Forbidden vocabulary list (no diagnosis words allowed)
- Lower temperature (0.1, set in gpt4o_client) reduces variance
- Format constraint forces structured output for post-hoc filtering

Prior work alignment:
- BiomedParse: GPT-4 used for label harmonization with ontology, not for
  ungrounded image description. We deviate for honesty.
- SkinCAP: dermatologist-authored captions DO see the diagnosis. Acceptable
  there (different use case: VLM pretraining, not closed-loop seg training).
- Standard practice in self-supervised captioning (BLIP, CLIP-Cap): zero
  external information given to the captioner.
"""

SYSTEM_PROMPT = """You are a dermatology vision assistant generating short, observation-only captions for dermoscopic skin lesion images. Each caption will serve as a text prompt for a downstream segmentation model. You must NOT have access to any ground truth, expert annotation, or patient information beyond what is visible in the image itself.

STRICT RULES
1. Output EXACTLY ONE sentence, between 15 and 25 words.
2. Begin with: "A dermoscopic image of"
3. End with a period.
4. Describe ONLY visual attributes that are clearly visible:
   - Lesion shape (symmetric / asymmetric / round / oval / irregular)
   - Border quality (regular / irregular / well-demarcated / notched / blurred)
   - Dominant colors (brown / black / red / blue / white / multi-colored)
   - Dermoscopic patterns IF obviously present (pigment network, dots, globules, streaks, blue-whitish veil, milia-like cysts, comedo-like openings, regression structures)
   - Approximate relative size (small / large lesion occupying most of the field)
5. FORBIDDEN — do NOT use any of these words or their variants:
   melanoma, nevus, mole, seborrheic, keratosis, benign, malignant, cancer, tumor, suspicious, biopsy, lesion type, diagnosis, atypical, dysplastic
6. Do NOT mention anatomic site (back, arm, torso, etc.) — you cannot reliably infer this from a close-up dermoscopic view.
7. Do NOT mention patient age or sex — these are not visible in the image.
8. If uncertain about a specific feature, OMIT it rather than guess. A shorter, accurate caption is better than a longer, hallucinated one.
9. Output exactly the sentence — no preamble like "Here is...", no markdown, no quotes.

EXAMPLE FORMATS (these are illustrative formats only, not tied to any specific image)
- A dermoscopic image of a small symmetric pigmented lesion with a regular brown network and well-demarcated border.
- A dermoscopic image of an asymmetric multi-colored lesion with irregular notched borders and a visible blue-whitish veil.
- A dermoscopic image of a well-demarcated brown lesion with milia-like cysts and multiple comedo-like openings across the surface.
- A dermoscopic image of a roughly oval pink-brown lesion with a faintly visible peripheral pigment network and regular borders.
- A dermoscopic image of an irregular dark lesion with multiple colors, asymmetric streaks at the periphery, and notched borders.
"""

USER_TEXT = """Generate the single-sentence caption for this image, following all rules strictly. Output only the caption sentence — nothing else."""


# Forbidden vocabulary used for post-hoc verification in evaluate.py.
# Any caption containing these tokens (case-insensitive, word-boundary) FAILS
# the no-leak check.
FORBIDDEN_TOKENS = {
    "melanoma", "melanocytic",
    "nevus", "nevi", "mole",
    "seborrheic", "keratosis", "keratoses",
    "benign", "malignant",
    "cancer", "carcinoma", "tumor", "tumour",
    "suspicious", "biopsy",
    "diagnosis", "diagnostic",
    "atypical", "dysplastic",
    "patient", "year-old", "year old",
    # anatomic sites — not visible from close-up dermoscopy
    "back", "torso", "abdomen", "chest", "arm", "leg", "thigh", "foot",
    "hand", "scalp", "face", "neck", "shoulder", "extremity",
}


def build_messages(image_b64_data_url: str) -> list[dict]:
    """Build the OpenAI chat messages payload for ONE ISIC image.

    No metadata, no ground truth — just the image and the system rules.

    Args:
        image_b64_data_url: data URL of the image, e.g.
            "data:image/jpeg;base64,/9j/4AAQ..."
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER_TEXT},
                {"type": "image_url",
                 "image_url": {"url": image_b64_data_url, "detail": "low"}},
            ],
        },
    ]
