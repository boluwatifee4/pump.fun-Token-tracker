#!/usr/bin/env python3
"""
🔎 Pump.fun Watcher – delayed-activation supervisor
• Runs token-fetcher on a schedule   (default: every 60 s)
• Polls each new mint’s bonding-curve every 30 s
• Spawns pf_tracker.py   *only when*  rSOL ≥ MIN_RAISED_SOL
"""

import asyncio
import subprocess
import csv
import os
import sys
import time
import logging
import aiohttp
import base64
import struct
from datetime import datetime, timezone
from typing import Dict, Set
from solana.publickey import PublicKey

# ───────── configurable ─────────
FETCH_INTERVAL = 60          # seconds between pf_tf.py runs
POLL_INTERVAL = 30          # seconds between curve polls
MIN_RAISED_SOL = 35          # start tracker above this
TRACKER_PATH = "pf_tracker.py"
FETCHER_PATH = "pf_tf.py"
TOKEN_CSV = "pump_tokens.csv"
HELIUS_KEY = os.getenv("HELIUS_KEY")
RPC_URL = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_KEY}"
PUMP_PROGRAM_ID = PublicKey("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")
BONDING_SEED = b"bonding-curve"

if not HELIUS_KEY:
    print("❌ HELIUS_KEY env var must be set")
    sys.exit(1)

# ───────── utilities ─────────


def curve_pda(mint: str) -> str:
    return str(PublicKey.find_program_address(
        [BONDING_SEED, bytes(PublicKey(mint))], PUMP_PROGRAM_ID)[0])


async def get_rsol(session: aiohttp.ClientSession, pda: str) -> float:
    payload = {"jsonrpc": "2.0", "id": 1, "method": "getAccountInfo",
               "params": [pda, {"encoding": "base64"}]}
    async with session.post(RPC_URL, json=payload) as r:
        j = await r.json()
    val = j.get("result", {}).get("value")
    if not val or not val.get("data"):          # unopened curve
        return 0.0
    buf = base64.b64decode(val["data"][0])
    if len(buf) >= 81:
        _, v_sol, _, r_sol, *_ = struct.unpack("<QQQQQ?32x", buf[8:81])
        return r_sol / 1e9
    return 0.0

# ───────── watcher class ─────────


class Watcher:
    def __init__(self):
        self.pending: Dict[str, str] = {}   # mint → pda
        self.active:  Dict[str, subprocess.Popen] = {}
        self.log = self._init_logger()

    def _init_logger(self):
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s | %(levelname)s | %(message)s",
                            handlers=[logging.FileHandler(f"logs/watch_{time.strftime('%Y%m%d')}.log"),
                                      logging.StreamHandler(sys.stdout)])
        return logging.getLogger("watcher")

    # ——— fetch new tokens every FETCH_INTERVAL ———
    async def _refresh_tokens(self):
        while True:
            self.log.info("📋 running token-fetcher …")
            proc = await asyncio.create_subprocess_exec(sys.executable, FETCHER_PATH)
            await proc.wait()

            added = 0
            if os.path.exists(TOKEN_CSV):
                with open(TOKEN_CSV, newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        mint = row["tokenAddress"]
                        if mint not in self.pending and mint not in self.active:
                            self.pending[mint] = curve_pda(mint)
                            added += 1
            self.log.info(f"➕ added {added} mints to watch-list")
            await asyncio.sleep(FETCH_INTERVAL)

    # ——— poll curves & spawn trackers ———
    async def _poll_curves(self):
        async with aiohttp.ClientSession() as sess:
            while True:
                for mint, pda in list(self.pending.items()):
                    try:
                        rsol = await get_rsol(sess, pda)
                        self.log.debug(f"{mint[:8]} rSOL={rsol:.2f}")
                        if rsol >= MIN_RAISED_SOL:
                            self._spawn_tracker(mint)
                            self.pending.pop(mint, None)
                            self.log.info(
                                f"🚀 {mint[:8]} raised {rsol:.1f} SOL – tracker started")
                    except Exception as e:
                        self.log.warning(f"⚠️  poll error {mint[:6]}…: {e}")
                        continue
                await asyncio.sleep(POLL_INTERVAL)

    # ——— watch running trackers & reap finished ———
    async def _reap_trackers(self):
        while True:
            for mint, proc in list(self.active.items()):
                if proc.poll() is not None:           # exited
                    self.log.info(f"✅ tracker finished {mint[:8]}")
                    self.active.pop(mint, None)
            await asyncio.sleep(15)

    def _spawn_tracker(self, mint: str):
        proc = subprocess.Popen([sys.executable, TRACKER_PATH, "--mint", mint])
        self.active[mint] = proc

    # ——— main entry ———
    async def run(self):
        await asyncio.gather(
            self._refresh_tokens(),
            self._poll_curves(),
            self._reap_trackers()
        )


# ───────── bootstrap ─────────
if __name__ == "__main__":
    try:
        asyncio.run(Watcher().run())
    except KeyboardInterrupt:
        print("\n⏹ stopped")
