#!/usr/bin/env python3
import asyncio
import subprocess
import time
import csv
import os
import sys
import logging
import aiohttp
from datetime import datetime, timezone, timedelta

# ───────── config ─────────
MORALIS_KEY = os.getenv("MORALIS_API_KEY")
POLL_SECONDS = 30
MINT_AGE_MIN = 45                 # ignore > 45-min mints
MAX_TRACKERS = 15                 # fits 50 req s-¹ quota
TRACK_SCRIPT = "pf2.py"           # single-mint tracker

# ───────── logger ─────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler(f"logs/orchestrator_{time.strftime('%Y%m%d')}.log"),
              logging.StreamHandler(sys.stdout)])
log = logging.getLogger("orc")

# ───────── Moralis fetch helper ─────────


async def fetch_new_mints(limit=100):
    url = f"https://solana-gateway.moralis.io/token/mainnet/exchange/pumpfun/new?limit={limit}"
    headers = {"X-API-Key": MORALIS_KEY, "accept": "application/json"}
    async with aiohttp.ClientSession() as s:
        async with s.get(url, headers=headers, timeout=8) as r:
            if r.status != 200:
                log.warning(f"Moralis HTTP {r.status}")
                return []
            data = (await r.json()).get("result", [])
            cutoff = datetime.utcnow().replace(tzinfo=timezone.utc) - \
                timedelta(minutes=MINT_AGE_MIN)
            return [t["tokenAddress"] for t in data
                    if datetime.fromisoformat(t["createdAt"].replace("Z", "+00:00")) >= cutoff]

# ───────── orchestrator loop ─────────


async def main():
    running = {}            # mint → Popen
    while True:
        try:
            # recycle finished trackers
            for m, p in list(running.items()):
                if p.poll() is not None:
                    log.info(f"💤 tracker finished {m[:8]}")
                    running.pop(m)

            # fetch latest mints
            new_mints = await fetch_new_mints()
            for mint in new_mints:
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

if __name__ == "__main__":
    if not MORALIS_KEY:
        log.error("Set MORALIS_API_KEY env var")
        sys.exit(1)
    asyncio.run(main())
