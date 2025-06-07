#!/usr/bin/env python3
import asyncio
import subprocess
import time
import os
import sys
import logging
import aiohttp
import re
from datetime import datetime, timezone, timedelta

# ───────── config ─────────
MORALIS_KEY = os.getenv("MORALIS_API_KEY")
POLL_SECONDS = 30
MINT_AGE_MIN = 45            # ignore > 45-min mints
MAX_TRACKERS = 15            # keeps Helius < 50 req s-¹
TRACK_SCRIPT = "pf2.py"      # single-mint tracker

# ───────── logger ─────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(f"logs/orchestrator_{time.strftime('%Y%m%d')}.log"),
              logging.StreamHandler(sys.stdout)])
log = logging.getLogger("orc")

# ───────── mint cleaner ─────────
BASE58_44 = re.compile(r"[1-9A-HJ-NP-Za-km-z]{44}")


def clean_mint(raw: str) -> str | None:
    """
    Moralis sometimes returns strings like
    'H48ZwFkzkR9UZULEnvHdXk8yss3LyqyLXMpgNRnbpump,'.
    Extract the **first** 44-char base-58 substring (a valid pubkey).
    Return None if none found.
    """
    m = BASE58_44.search(raw)
    return m.group(0) if m else None

# ───────── Moralis fetch helper ─────────


async def fetch_new_mints(limit: int = 100) -> list[str]:
    url = f"https://solana-gateway.moralis.io/token/mainnet/exchange/pumpfun/new?limit={limit}"
    headers = {"X-API-Key": MORALIS_KEY, "accept": "application/json"}
    async with aiohttp.ClientSession() as sess:
        async with sess.get(url, headers=headers, timeout=8) as r:
            if r.status != 200:
                log.warning(f"Moralis HTTP {r.status}")
                return []
            data = (await r.json()).get("result", [])
            cutoff = datetime.now(timezone.utc) - \
                timedelta(minutes=MINT_AGE_MIN)
            return [
                t["tokenAddress"]
                for t in data
                if datetime.fromisoformat(t["createdAt"].replace("Z", "+00:00")) >= cutoff
            ]

# ───────── orchestrator loop ─────────


async def main():
    running: dict[str, subprocess.Popen] = {}

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
                if mint is None:
                    log.warning(f"Skip malformed mint: {raw}")
                    continue
                if mint in running or len(running) >= MAX_TRACKERS:
                    continue

                log.info(f"🚀 spawn tracker {mint[:8]}")
                proc = subprocess.Popen(
                    [sys.executable, TRACK_SCRIPT, "--mint", mint])
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
    asyncio.run(main())
