#!/usr/bin/env python3
"""Make one real chat completion call through a running proxy.

Holds the real credential: `start_proxy.py` only relays whatever
`Authorization` it receives, so the client is the one that must supply a
real provider key. The only thing borrowed from the proxy side is its URL
(`PROXY_URL`, must match `start_proxy.py`'s host/port).

Setup::

    cp .env.example .env
    # edit .env: set MODEL_API_KEY for your provider

Run (in a second terminal, after ``start_proxy.py`` is up)::

    uv run python scripts/call_model.py
"""

from __future__ import annotations

import os
from pathlib import Path

from openai import OpenAI

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
PROXY_URL = "http://127.0.0.1:8901/v1"  # must match start_proxy.py's host/port
MODEL_NAME = "deepseek-v4-flash"


def load_dotenv(path: Path) -> None:
    """Populate ``os.environ`` from a plain ``KEY=VALUE`` .env file."""
    if not path.exists():
        raise SystemExit(f"missing {path} — copy .env.example to .env and fill in your provider's details")
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def call_once(client: OpenAI) -> None:
    """Non-streaming call: the whole reply arrives in one response object."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Do Fava proxy work?"}],
    )
    print(f"model replied: {response.choices[0].message.content!r}")


def call_streaming(client: OpenAI) -> None:
    """Streaming call: the reply arrives as SSE chunks the proxy relays live."""
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Count from 1 to 5, one number per line. wait at least 1 sec between each number."}],
        stream=True,
    )
    print("streaming reply: ", end="", flush=True)
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            print(delta, end="", flush=True)
    print()


def main() -> int:
    load_dotenv(ENV_PATH)
    model_api_key = os.environ["MODEL_API_KEY"]
    client = OpenAI(base_url=PROXY_URL, api_key=model_api_key)

    call_once(client)
    call_streaming(client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
