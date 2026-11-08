"""Smoke test for step 3: POST /verify on a few answers (live Gemini via Bifrost + live Tavily).

Spends ~1 Tavily credit per volatile claim. Usage (StaleGuard running on :8000):
    uv run python scripts/smoke_verify.py
    uv run python scripts/smoke_verify.py --text "Your answer text here"
"""

import argparse
import sys

import httpx

SAMPLES = [
    ("stale-prone", "What's the latest Python?", "The latest stable version of Python is 3.12, released in October 2023."),
    ("static", None, "Water boils at 100 degrees Celsius at sea level, and a triangle has three sides."),
    ("mixed", "Tell me about OpenAI.", "OpenAI was founded in 2015. Its CEO is Sam Altman. GPT-4 is its newest model."),
]

COLOR = {"CURRENT": "\033[32m", "STALE": "\033[31m", "UNVERIFIABLE": "\033[33m", "CONFLICTING": "\033[35m"}
RESET = "\033[0m"


def show(label: str, report: dict) -> None:
    print(f"\n=== {label}: {report['verdict_summary'].upper()}  "
          f"(extracted {report['claims_extracted']}, checked {report['claims_checked']}, "
          f"tavily {report['tavily_calls']}, {report['latency_ms']} ms, tokens {report['tokens_by_stage']})")
    for v in report["claims"]:
        c = v["claim"]
        print(f"  {COLOR.get(v['verdict'], '')}{v['verdict']:12}{RESET} [{c['claim_class']}/{c['classified_by']}] "
              f"conf={v['confidence']:.2f}  {c['text']}")
        if v["corrected_claim"]:
            print(f"      -> {v['corrected_claim']}")
        print(f"      {v['rationale']}")
        for e in v["evidence"][:2]:
            print(f"      - {e['published_date'] or '????-??-??'}  {e['url']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/verify")
    parser.add_argument("--text")
    args = parser.parse_args()
    samples = [("custom", None, args.text)] if args.text else SAMPLES
    for label, question, text in samples:
        r = httpx.post(args.url, json={"text": text, "question": question}, timeout=180)
        if r.status_code != 200:
            print(f"\n=== {label}: HTTP {r.status_code} {r.text[:500]}")
            return 1
        show(label, r.json())
    return 0


if __name__ == "__main__":
    sys.exit(main())
