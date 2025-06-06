#!/usr/bin/env python3
"""
📡 Pump.fun Tracker – dual-CSV, single-mint edition
• Candle rows every 1/5/10 s  ➜ all_token_tracks.csv
• One row per on-chain event ➜ all_token_events.csv
"""

import asyncio
import aiohttp
import csv
import os
import sys
import time
import base64
import struct
import logging
import threading
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from solana.publickey import PublicKey   # pip install solana

# ───────── CLI argument (one mint per process) ─────────
import argparse
ap = argparse.ArgumentParser(description="Track a single Pump.fun mint")
ap.add_argument("--mint", required=True, help="Token mint address")
ARGS = ap.parse_args()

# ───────── constants ─────────
HELIUS_KEY = os.getenv("HELIUS_KEY")
if not HELIUS_KEY:
    print("❌ HELIUS_KEY env var not set")
    sys.exit(1)

RPC = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_KEY}"
WS = f"wss://mainnet.helius-rpc.com/?api-key={HELIUS_KEY}"

TRACKS_CSV = "all_token_tracks.csv"
EVENTS_CSV = "all_token_events.csv"

PUMP_PROGRAM_ID = PublicKey("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")
BONDING_SEED = b"bonding-curve"
MIGRATION_HELPER = "39azUYFWPz3VHgKCf3VChUwbpURdCHRxjWVowf5jUJjg"

INITIAL_RTOK_RES = 793_100_000 * 1_000_000
DURATION_SECS = 45 * 60
MIN_VOL_SOL = 0.00005

TRACK_HEADER = [
    "timestamp", "mint", "sol_in_pool", "sol_raised_total", "progress",
    "sol_flow", "sol_accel", "unique_buyers", "top3_pct", "lp_burn",
    "whale_flag", "buy_sell_delta", "buy_pressure", "sell_pressure", "bot_like"
]

EVENT_HEADER = [
    "slot", "block_time", "recv_time", "mint", "signature",
    "is_buy", "amount_sol", "v_sol", "v_tok", "buyer_pubkey"
]

# ───────── helpers ─────────


def now_iso(): return datetime.now(timezone.utc).isoformat()


def curve_pda(mint: str) -> str:
    return str(PublicKey.find_program_address(
        [BONDING_SEED, bytes(PublicKey(mint))], PUMP_PROGRAM_ID)[0])

# ───────── dataclasses ─────────


@dataclass
class IntervalState:
    start: datetime
    last_R: float = 0.
    last_sol: float = 0.


@dataclass
class SamplerTask:
    mint: str
    name: str
    launch: datetime
    curve: str
    state: IntervalState
    lp_mint: Optional[str] = None
    migration_event: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class Tx:
    sig: str
    signer: str
    sol: float
    ts: int
    is_buy: bool

# ───────── tracker ─────────


class Tracker:
    def __init__(self):
        self.log = self._init_logger()
        self.session: Optional[aiohttp.ClientSession] = None
        self.csv_lock = threading.Lock()
        self.active: Dict[str, SamplerTask] = {}

        for fn, h in [(TRACKS_CSV, TRACK_HEADER), (EVENTS_CSV, EVENT_HEADER)]:
            if not os.path.exists(fn):
                with open(fn, "w", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=h).writeheader()

    def _init_logger(self):
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s | %(levelname)s | %(message)s",
                            handlers=[logging.FileHandler(f"logs/tracker_{time.strftime('%Y%m%d')}.log"),
                                      logging.StreamHandler(sys.stdout)])
        return logging.getLogger("tracker")

    # ─────── main entry ───────
    async def run(self):
        async with aiohttp.ClientSession() as sess:
            self.session = sess
            self.log.info("RPC ready")

            # build metadata for the single mint supplied on CLI
            mint_meta = {
                "tokenAddress": ARGS.mint,
                "name": ARGS.mint[:8],
                "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "trackable": True,
            }
            await self._start(mint_meta)

            await asyncio.gather(
                self._monitor(), self._heartbeat(), self._event_hub(),
                return_exceptions=True)

    # ─────── bootstrap one sampler ───────
    async def _start(self, row):
        mint, rowname = row["tokenAddress"], row.get("name", "?")
        launch = datetime.fromisoformat(
            row["createdAt"].replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc)-launch).total_seconds()
        if age > DURATION_SECS:
            return self.log.info(f"{rowname} too old, skip")

        st = SamplerTask(mint, rowname, launch, curve_pda(mint),
                         IntervalState(datetime.now(timezone.utc)))
        self.active[mint] = st
        asyncio.create_task(self._sampler(st))
        self.log.info(f"📡 {rowname[:12]}… sampler started")

    # ─────── sampler (candle writer) ───────
    async def _sampler(self, st: SamplerTask):
        while not st.migration_event.is_set():
            age = (datetime.now(timezone.utc)-st.launch).total_seconds()
            prog = await self._progress_pct(st)
            window = 1 if (age < 120 or prog < 90) else 5 if age < 600 else 10
            try:
                row = await self._collect(st)
                self._csv_write(TRACKS_CSV, row, TRACK_HEADER)
            except Exception as e:
                self.log.error(f"{st.name} collect fail: {e}")
            await asyncio.sleep(window)
        self.active.pop(st.mint, None)
        self.log.info(f"✅ sampler done {st.name}")

    # ─────── metric collection ───────
    async def _collect(self, st):
        curve = await self._get_curve(st.curve)
        v_sol, v_tok, r_sol = curve["v_sol"], curve["v_tok"], curve["r_sol"]
        sol_pool, raised = v_sol/1e9, r_sol/1e9
        prog = max(0, min(100, 100-(curve["r_tok"]/INITIAL_RTOK_RES)*100))

        flow = raised - st.state.last_sol
        accel = flow - st.state.last_R
        st.state.last_sol, st.state.last_R = raised, flow

        txs = await self._recent_txs(st.curve)
        tm = self._tx_metrics(txs)
        delta = tm["buy_vol"]-tm["sell_vol"]

        return dict(timestamp=now_iso(), mint=st.mint,
                    sol_in_pool=round(sol_pool, 6), sol_raised_total=round(raised, 6),
                    progress=round(prog, 2), sol_flow=round(flow, 6), sol_accel=round(accel, 6),
                    unique_buyers=tm["unique"], top3_pct=round(
                        tm["top3_pct"], 2),
                    lp_burn=False, whale_flag=tm["whale"], buy_sell_delta=round(delta, 4),
                    buy_pressure=round(tm["buy_vol"], 5), sell_pressure=round(tm["sell_vol"], 5),
                    bot_like=tm["bot"])

    async def _progress_pct(self, st):
        curve = await self._get_curve(st.curve)
        return max(0, min(100, 100-(curve["r_tok"]/INITIAL_RTOK_RES)*100))

    # ─────── curve state ───────
    async def _get_curve(self, addr):
        payload = {"jsonrpc": "2.0", "id": 1, "method": "getAccountInfo",
                   "params": [addr, {"encoding": "base64"}]}
        val = (await self._rpc(payload)).get("result", {}).get("value")
        if not val or not val.get("data"):
            return dict(v_tok=1, v_sol=0, r_tok=INITIAL_RTOK_RES, r_sol=0)
        buf = base64.b64decode(val["data"][0])
        if len(buf) >= 81:
            v_tok, v_sol, r_tok, r_sol, * \
                _ = struct.unpack("<QQQQQ?32x", buf[8:81])
            return dict(v_tok=v_tok, v_sol=v_sol, r_tok=r_tok, r_sol=r_sol)
        return dict(v_tok=1, v_sol=0, r_tok=INITIAL_RTOK_RES, r_sol=0)

    # ─────── transaction helpers ───────
    async def _recent_txs(self, curve):
        payload = {"jsonrpc": "2.0", "id": 1, "method": "getSignaturesForAddress",
                   "params": [curve, {"limit": 200}]}
        sigs = [s["signature"] for s in (await self._rpc(payload)).get("result", [])]
        out = []
        for sig in sigs:
            tx = await self._get_tx(sig)
            if tx:
                out.append(tx)
        return out

    async def _get_tx(self, sig):
        payload = {"jsonrpc": "2.0", "id": 1, "method": "getTransaction",
                   "params": [sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}]}
        data = (await self._rpc(payload)).get("result")
        if not data or data.get("meta", {}).get("err"):
            return None
        msg, pre, post = data["transaction"]["message"], data["meta"]["preBalances"], data["meta"]["postBalances"]
        signer_idx = next((i for i, a in enumerate(
            msg["accountKeys"]) if a.get("signer")), None)
        if signer_idx is None:
            return None
        sol = (post[signer_idx]-pre[signer_idx])/1e9
        if abs(sol) < MIN_VOL_SOL:
            return None
        return Tx(sig, msg["accountKeys"][signer_idx]["pubkey"], abs(sol), data["blockTime"], sol < 0)

    def _tx_metrics(self, txs: List[Tx]):
        buy = sell = 0
        buyers = {}
        for t in txs:
            if t.is_buy:
                buy += t.sol
                buyers[t.signer] = buyers.get(t.signer, 0)+t.sol
            else:
                sell += t.sol
        top3 = sum(sorted(buyers.values(), reverse=True)
                   [:3])/buy*100 if buy else 0
        return dict(buy_vol=buy, sell_vol=sell, unique=len(buyers),
                    top3_pct=top3, whale=any(t.sol >= 1 for t in txs), bot=False)

    # ─────── WebSocket event hub (single curve) ───────
    async def _event_hub(self):
        """
        Streams every on-chain swap that touches *this* bonding-curve PDA
        and appends one row per tx to all_token_events.csv
        """
        if not self.active:
            return

        st = next(iter(self.active.values()))      # the only SamplerTask
        pda = st.curve                              # PDA string

        async with self.session.ws_connect(WS) as ws:
            # 1️⃣  SUBSCRIBE  –  Helius 'transactionSubscribe'
            await ws.send_json({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "transactionSubscribe",
                "params": [
                    # any tx whose *message* mentions our PDA
                    {"mentions": [pda]},
                    {
                        "commitment":        "confirmed",
                        "encoding":          "jsonParsed",
                        "transactionDetails": "full"     # we want pre/post balances
                    }
                ]
            })

            # 2️⃣  LISTEN
            async for msg in ws:
                if msg.type is not aiohttp.WSMsgType.TEXT:
                    break

                j = msg.json()
                if j.get("method") != "transactionNotification":
                    # ping, error, or unrelated subscription
                    continue

                tx_obj = j["params"]["result"]["transaction"]
                slot = j["params"]["result"]["context"]["slot"]
                block_time = j["params"]["result"]["blockTime"]

                # sanity-check: does the PDA really appear in the account list?
                if pda not in (a if isinstance(a, str) else a.get("pubkey")
                               for a in tx_obj["message"]["accountKeys"]):
                    continue

                # signer & SOL diff
                acct_keys = tx_obj["message"]["accountKeys"]
                signer_idx = next((i for i, acc in enumerate(acct_keys)
                                   if acc.get("signer")), None)
                if signer_idx is None:
                    continue

                pre = tx_obj["meta"]["preBalances"][signer_idx]
                post = tx_obj["meta"]["postBalances"][signer_idx]
                sol_diff = (post - pre) / 1e9

                signer_pub = acct_keys[signer_idx]["pubkey"]

                # reserves *after* the tx
                curve_state = await self._get_curve(pda)
                v_sol = curve_state["v_sol"] / 1e9
                v_tok = curve_state["v_tok"] / 1e9

                row = dict(
                    slot=slot,
                    block_time=block_time,
                    recv_time=now_iso(),
                    mint=st.mint,
                    signature=tx_obj["signatures"][0],
                    is_buy=sol_diff < 0,
                    amount_sol=round(abs(sol_diff), 6),
                    v_sol=round(v_sol, 6),
                    v_tok=round(v_tok, 6),
                    buyer_pubkey=signer_pub,
                )
                self._csv_write(EVENTS_CSV, row, EVENT_HEADER)

    # ─────── misc utils ───────
    def _csv_write(self, fn, row, header):
        with self.csv_lock:
            with open(fn, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=header).writerow(row)

    async def _rpc(self, payload, retries=3):
        for a in range(retries):
            try:
                async with self.session.post(RPC, json=payload, timeout=8) as r:
                    j = await r.json()
                    err = j.get("error")
                    if err:
                        raise RuntimeError(err.get("message", "RPC"))
                    return j
            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError):
                if a == retries-1:
                    raise
                await asyncio.sleep(0.4*2**a)

    async def _monitor(self):
        while self.active:
            await asyncio.sleep(30)
        self.log.info("🎬 all samplers finished")

    async def _heartbeat(self):
        while True:
            self.log.info(f"⏰ heartbeat | active={len(self.active)}")
            await asyncio.sleep(300)


# ───────── main ─────────
if __name__ == "__main__":
    try:
        asyncio.run(Tracker().run())
    except KeyboardInterrupt:
        print("\n⏹ stopped")
