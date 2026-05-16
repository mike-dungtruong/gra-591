"""
Thin wrapper around OpenAI Chat Completions for vision captioning.

Features:
- Retries with exponential backoff for transient errors / rate limits.
- Token + cost accounting (GPT-4o pricing as of 2026-05).
- Hard timeout per call so a hung request doesn't block the pilot.
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass

# openai is imported lazily inside GPT4oClient.__init__ so that importing
# this module (e.g. for CaptionResult) does not require openai to be installed.

# Pricing in USD per 1M tokens. Update if OpenAI changes pricing.
# Source: https://openai.com/api/pricing/ (verify before billing analysis).
PRICING = {
    "gpt-4o":         {"in": 2.50, "out": 10.00},
    "gpt-4o-mini":    {"in": 0.15, "out": 0.60},
    "gpt-4.1":        {"in": 2.00, "out": 8.00},
    "gpt-4.1-mini":   {"in": 0.40, "out": 1.60},
}


@dataclass
class CaptionResult:
    image_id: str
    caption: str
    raw_response: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_s: float
    error: str | None = None


class GPT4oClient:
    def __init__(self, model: str = "gpt-4o",
                 timeout: float = 60.0,
                 max_retries: int = 6):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for caption generation. "
                "Run: pip install 'openai>=1.40.0'"
            )
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Run:\n"
                "    export OPENAI_API_KEY='sk-...'"
            )
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.model = model
        self.max_retries = max_retries
        if model not in PRICING:
            print(f"[warn] no pricing entry for {model}, cost will be 0")

    def _cost(self, in_toks: int, out_toks: int) -> float:
        p = PRICING.get(self.model, {"in": 0, "out": 0})
        return (in_toks * p["in"] + out_toks * p["out"]) / 1_000_000

    def caption(self, image_id: str, messages: list[dict],
                max_tokens: int = 80, temperature: float = 0.3) -> CaptionResult:
        """Call GPT-4o once. Retries on transient errors.

        max_tokens=80 is enough for our 25-word target (~1.3 token/word).
        temperature=0.3 keeps output stable but allows minor lexical variation
        (helps lexical diversity metric).
        """
        from openai import APIError, RateLimitError, APITimeoutError

        delay = 2.0
        last_err: str | None = None

        for attempt in range(1, self.max_retries + 1):
            t0 = time.time()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore[arg-type]
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                latency = time.time() - t0
                txt = resp.choices[0].message.content or ""
                in_toks = resp.usage.prompt_tokens if resp.usage else 0
                out_toks = resp.usage.completion_tokens if resp.usage else 0
                return CaptionResult(
                    image_id=image_id,
                    caption=txt.strip(),
                    raw_response=txt,
                    model=self.model,
                    prompt_tokens=in_toks,
                    completion_tokens=out_toks,
                    cost_usd=self._cost(in_toks, out_toks),
                    latency_s=latency,
                )
            except RateLimitError as e:
                last_err = f"RateLimitError: {e}"
                if attempt == self.max_retries:
                    break
                # Honour Retry-After if present; otherwise floor at 60s.
                retry_after = None
                if hasattr(e, "response") and e.response is not None:
                    retry_after = e.response.headers.get("retry-after")
                wait = (float(retry_after) if retry_after
                        else max(delay, 60.0)) + random.uniform(0, 5)
                print(f"  [retry {attempt}/{self.max_retries}] rate limited — "
                      f"sleeping {wait:.1f}s")
                time.sleep(wait)
                delay = min(delay * 2, 120.0)
            except APITimeoutError as e:
                last_err = f"APITimeoutError: {e}"
                if attempt == self.max_retries:
                    break
                wait = delay + random.uniform(0, 1)
                print(f"  [retry {attempt}/{self.max_retries}] timeout — "
                      f"sleeping {wait:.1f}s")
                time.sleep(wait)
                delay = min(delay * 2, 60.0)
            except APIError as e:
                status = getattr(e, "status_code", None)
                last_err = f"APIError({status}): {e}"
                if status in {400, 401, 403, 413}:
                    break  # non-retriable
                if attempt == self.max_retries:
                    break
                wait = delay + random.uniform(0, 1)
                print(f"  [retry {attempt}/{self.max_retries}] API error {status} — "
                      f"sleeping {wait:.1f}s")
                time.sleep(wait)
                delay = min(delay * 2, 60.0)
            except Exception as e:  # noqa: BLE001
                last_err = f"{type(e).__name__}: {e}"
                break  # unknown — don't retry

        return CaptionResult(
            image_id=image_id,
            caption="",
            raw_response="",
            model=self.model,
            prompt_tokens=0,
            completion_tokens=0,
            cost_usd=0.0,
            latency_s=0.0,
            error=last_err,
        )
