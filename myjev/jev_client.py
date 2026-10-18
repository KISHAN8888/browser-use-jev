"""Minimal TypeSafe client with a persistent spend ledger and a hard budget cap.

Every call is appended to data/jev_calls.jsonl (request id, usage, cost, latency). Before each call
the ledger total is checked against the cap, so a crashed or resumed run can never overspend.
"""

import json
import os
import threading
import time
from pathlib import Path

import httpx

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
PRICE_PER_M_INPUT = 0.042  # USD, docs.typesafe.ai/models.md (output tokens are free)
LEDGER = Path(os.environ.get("MYJEV_JEV_LEDGER", "data/jev_calls.jsonl"))
LEDGER_LOCK = threading.Lock()  # concurrent workers must not interleave ledger lines


def load_env(path=".env"):
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                if value.strip():
                    os.environ.setdefault(key.strip(), value.strip())


def spent():
    if not LEDGER.exists():
        return 0.0
    with LEDGER_LOCK:
        lines = LEDGER.read_text(encoding="utf-8").splitlines()
    total = 0.0
    for line in lines:
        try:
            total += json.loads(line)["cost_usd"]
        except (json.JSONDecodeError, KeyError):
            raise RuntimeError(f"Ledger {LEDGER} has an unreadable line; fix it before spending more.") from None
    return total


def estimate_tokens(body):
    # ~3.5 chars/token: measured 2,304 real tokens vs 2,014 at 4 chars/token on the smoke request.
    return int(len(json.dumps(body, ensure_ascii=False)) / 3.5)


class Jev:
    def __init__(self, budget_usd, model=None):
        load_env()
        self.key = os.environ["TYPESAFE_API_KEY"]
        self.model = model or os.environ.get("TYPESAFE_MODEL") or "jev-latest"
        self.budget = budget_usd
        self.local = threading.local()  # one HTTP client per worker thread (a shared one fails on Windows)
        LEDGER.parent.mkdir(parents=True, exist_ok=True)

    @property
    def client(self):
        if not hasattr(self.local, "client"):
            self.local.client = httpx.Client(http2=False, timeout=60)
        return self.local.client

    def ask(self, body, tag=""):
        body = {**body, "model": self.model}
        projected = spent() + estimate_tokens(body) * PRICE_PER_M_INPUT / 1e6
        if projected > self.budget:
            raise RuntimeError(f"Budget cap ${self.budget:.4f} would be exceeded (ledger ${spent():.4f}); not calling.")
        for attempt in range(5):
            started = time.perf_counter()
            try:
                response = self.client.post(ENDPOINT, json=body, headers={"Authorization": f"Bearer {self.key}"})
            except httpx.TransportError as error:
                self.local.__dict__.pop("client", None)  # drop the broken connection
                if attempt == 4:
                    raise RuntimeError(f"transport error (not a Jev answer): {error!r}") from None
                time.sleep(0.5 * 2**attempt)
                continue
            latency = round((time.perf_counter() - started) * 1000)
            if response.status_code in {429, 500, 502, 503, 529} and attempt < 4:
                time.sleep(0.5 * 2**attempt)
                continue
            break
        result = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        usage = result.get("usage", {}) if isinstance(result, dict) else {}
        input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens") or estimate_tokens(body)
        entry = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": tag, "status": response.status_code,
            "latency_ms": latency, "usage": usage, "input_tokens": input_tokens,
            "usage_estimated": not usage, "cost_usd": input_tokens * PRICE_PER_M_INPUT / 1e6,
        }
        with LEDGER_LOCK, LEDGER.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        if response.is_error:
            raise RuntimeError(f"TypeSafe HTTP {response.status_code}: {response.text[:300]}")
        return result, entry
