#!/usr/bin/env python3
"""
orchestrator.py – identical flow, but TLS-safe.

Fix: use certifi-backed SSL context so macOS / custom Python builds
stop raising SSLCertVerificationError when talking to Moralis.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
import os
import sys
import logging
import aiohttp
import certifi
import ssl
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

# ───────── config ─────────
MORALIS_KEY = os.getenv("MORALIS_API_KEY")
POLL_SECONDS = 30
MINT_AGE_MIN = 45           # ignore > 45-min mints
MAX_TRACKERS = 15           # keeps Helius < 50 req s-¹
TRACK_SCRIPT = "pf2.py"

# ───────── logger ─────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(f"logs/orchestrator_{time.strftime('%Y%m%d')}.log",
                            encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("orc")

# ───────── base-58 regex ─────────
BASE58_44 = re.compile(r"[1-9A-HJ-NP-Za-km-z]{44}")

# ───────── ssl helper ─────────
def ssl_connector() -> aiohttp.TCPConnector:
    ctx = ssl.create_default_context(cafile=certifi.where())
    return aiohttp.TCPConnector(ssl=ctx)

# ───────── misc helpers ─────────
def clean_mint(raw: str) -> Optional[str]:
    m = BASE58_44.search(raw)
    return m.group(0) if m else None


async def fetch_new_mints(limit: int = 100) -> List[str]:
    url = (
        f"https://solana-gateway.moralis.io/token/mainnet/exchange/pumpfun/new?limit={limit}"
    )
    headers = {"X-API-Key": MORALIS_KEY, "accept": "application/json"}
    async with aiohttp.ClientSession(
        connector=ssl_connector(), timeout=aiohttp.ClientTimeout(total=8)
    ) as sess:
        async with sess.get(url, headers=headers) as r:
            if r.status != 200:
                log.warning(f"Moralis HTTP {r.status}")
                return []
            data = (await r.json()).get("result", [])
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=MINT_AGE_MIN)
            return [
                t["tokenAddress"]
                for t in data
                if datetime.fromisoformat(t["createdAt"].replace("Z", "+00:00"))
                >= cutoff
            ]


# ───────── orchestrator main loop ─────────
async def main() -> None:
    running: Dict[str, subprocess.Popen] = {}

    while True:
        try:
            # reap finished trackers
            for mint, proc in list(running.items()):
                if proc.poll() is not None:
                    log.info(f"💤 tracker finished {mint[:8]}")
                    running.pop(mint)

            # fetch latest launches
            for raw in await fetch_new_mints():
                mint = clean_mint(raw)
                if (
                    not mint
                    or mint in running
                    or len(running) >= MAX_TRACKERS
                ):
                    continue

                log.info(f"🚀 spawn tracker {mint[:8]}")
                proc = subprocess.Popen(
                    [sys.executable, TRACK_SCRIPT, "--mint", mint]
                )
                running[mint] = proc

            log.info(f"📊 active trackers: {len(running)}")
            await asyncio.sleep(POLL_SECONDS)

        except KeyboardInterrupt:
            log.info("⏹ stopping orchestrator")
            break
        except Exception as e:
            log.error(f"loop error: {e}")
            await asyncio.sleep(5)

# ───────── bootstrap ─────────
if __name__ == "__main__":
    if not MORALIS_KEY:
        log.error("Set MORALIS_API_KEY env var")
        sys.exit(1)

    # ensure certifi present
    try:
        import certifi  # noqa: F401
    except ImportError:
        log.error("Run `pip install certifi` to fix TLS certificates")
        sys.exit(1)

    asyncio.run(main())
