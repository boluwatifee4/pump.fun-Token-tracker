#!/usr/bin/env python3
"""
📡 Pump.fun Tracker – dual-CSV, single-mint edition
• Candle rows every 1/5/10 s  ➜ all_token_tracks.csv
• One row per on-chain event ➜ all_token_events.csv

Includes agnostic migration detection:
• PDA missing/closed OR Raydium AMM detected
• Fires off a migrated_tracker.py subprocess on migrate
• Filters out unsafe tokens via RugCheck API
"""

import asyncio
import aiohttp
import ssl, certifi
import csv
import os
import sys
import time
import base64
import struct
import logging
import threading
import subprocess
import requests  # Added for RugCheck
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from solana.publickey import PublicKey

# ───────── TLS context ─────────
SSL_CTX = ssl.create_default_context(cafile=certifi.where())

# ───────── CLI args ─────────
import argparse
ap = argparse.ArgumentParser(description="Track a single Pump.fun mint")
ap.add_argument("--mint", required=True, help="Token mint address")
ap.add_argument("--sampler-ms", type=int, help="Sampler interval in ms")
ap.add_argument("--max-age-h", type=float, default=4)
ap.add_argument("--debug-migrate", action="store_true", help="Simulate migration after 30s")
ap.add_argument("--debug-launch", action="store_true", help="Launch migrated tracker without migration")
ARGS = ap.parse_args()

# ───────── RugCheck setup ─────────
RUGCHECK_API_KEY = os.getenv("RUGCHECK_API_KEY")
_last_rugcheck_call = 0
_min_rugcheck_interval = 2.0  # Increased to 2 seconds minimum
_rugcheck_cache: Dict[str, Dict] = {}

def is_token_safe(mint: str) -> bool:
    """
    Check token safety criteria:
    1. 100% LP must be locked
    2. Mint authority must be revoked
    3. Freeze authority must be revoked
    """
    global _last_rugcheck_call, _min_rugcheck_interval
    
    if not RUGCHECK_API_KEY:
        return True
    
    # Check cache first
    if mint in _rugcheck_cache:
        cached_result = _rugcheck_cache[mint]
        cache_age = time.time() - cached_result.get('timestamp', 0)
        if cache_age < 3600:  # Cache for 1 hour
            print(f"🔄 Using cached RugCheck result for {mint[:8]}...")
            return cached_result['is_safe']
    
    # Rate limiting
    current_time = time.time()
    time_since_last = current_time - _last_rugcheck_call
    if time_since_last < _min_rugcheck_interval:
        sleep_time = _min_rugcheck_interval - time_since_last
        print(f"⏳ Rate limiting: waiting {sleep_time:.1f}s...")
        time.sleep(sleep_time)
    
    # Use full report endpoint instead of summary
    url = f"https://api.rugcheck.xyz/v1/tokens/{mint}/report"
    headers = {
        # "X-API-KEY": RUGCHECK_API_KEY,
        "Accept": "application/json"
    }
    
    try:
        _last_rugcheck_call = time.time()
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        report = r.json()
        
        is_safe = True
        rejection_reasons = []
        
        # 1. Check mint authority
        mint_authority = report.get("mintAuthority")
        if mint_authority and mint_authority != "11111111111111111111111111111111":
            rejection_reasons.append("Mint authority not revoked")
            is_safe = False
            
        # 2. Check freeze authority
        freeze_authority = report.get("freezeAuthority")
        if freeze_authority and freeze_authority != "11111111111111111111111111111111":
            rejection_reasons.append("Freeze authority not revoked")
            is_safe = False
            
        # 3. Check LP lock percentage (must be 100%)
        lp_locked = report.get("lpLockedPct", 0)
        if lp_locked < 100:
            rejection_reasons.append(f"LP not 100% locked (only {lp_locked}%)")
            is_safe = False
        
        # Cache result
        _rugcheck_cache[mint] = {
            'is_safe': is_safe,
            'timestamp': time.time(),
            'reasons': rejection_reasons,
            'lp_locked': lp_locked,
            'mint_auth': bool(mint_authority),
            'freeze_auth': bool(freeze_authority)
        }
        
        if is_safe:
            print(f"✅ Token {mint[:8]}... passed ALL security checks:")
            print(f"   • Mint authority: Revoked")
            print(f"   • Freeze authority: Revoked") 
            print(f"   • LP 100% locked")
        else:
            print(f"❌ Token {mint[:8]}... failed: {', '.join(rejection_reasons)}")
            
        return is_safe
        
    except Exception as e:
        print(f"⚠️ RugCheck error: {e}")
        return True

def check_rugcheck_status():
    """Check if RugCheck API is accessible and what rate limits apply"""
    if not RUGCHECK_API_KEY:
        print("❌ No RugCheck API key configured")
        return False
    
    try:
        r = requests.get("https://api.rugcheck.xyz/utils/chains", 
                        headers={"X-API-KEY": RUGCHECK_API_KEY}, timeout=10)
        if r.status_code == 200:
            print("✅ RugCheck API accessible")
            if 'X-RateLimit-Remaining' in r.headers:
                print(f"📊 Rate limit remaining: {r.headers['X-RateLimit-Remaining']}")
            return True
        else:
            print(f"⚠️ RugCheck API returned status {r.status_code}")
            return False
    except Exception as e:
        print(f"❌ RugCheck API not accessible: {e}")
        return False

# ───────── constants ─────────
HELIUS_KEY = os.getenv("HELIUS_KEY")
if not HELIUS_KEY:
    print("❌ HELIUS_KEY env var not set")
    sys.exit(1)

RPC = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_KEY}"
WS  = f"wss://mainnet.helius-rpc.com/?api-key={HELIUS_KEY}"

TRACKS_CSV = "all_token_tracks.csv"
EVENTS_CSV = "all_token_events.csv"

PUMP_PROGRAM_ID = PublicKey("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")
BONDING_SEED    = b"bonding-curve"

MIGRATION_ROUTER = "39azUYFWPz3VHgKCf3VChUwbpURdCHRxjWVowf5jUJjg"
RAYDIUM_AMM     = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
INIT_OPCODE     = "initialize2"

INITIAL_RTOK_RES = 793_100_000 * 1_000_000
DURATION_SECS    = 45 * 60
MIN_VOL_SOL      = 0.00005

MIGRATED_TRACKS_DIR = "migrated_tracks"
os.makedirs(MIGRATED_TRACKS_DIR, exist_ok=True)


TRACK_HEADER = [
    "timestamp", "mint", "sol_in_pool", "sol_raised_total", "progress",
    "sol_flow", "sol_accel", "unique_buyers", "top3_pct", "lp_burn",
    "whale_flag", "buy_sell_delta", "buy_pressure", "sell_pressure", "bot_like",
    "migrated"
]

EVENT_HEADER = [
    "slot", "block_time", "recv_time", "mint", "signature",
    "is_buy", "amount_sol", "v_sol", "v_tok", "buyer_pubkey",
    "migrated"
]

def now_iso(): return datetime.now(timezone.utc).isoformat()

def curve_pda(mint: str) -> str:
    return str(PublicKey.find_program_address(
        [BONDING_SEED, bytes(PublicKey(mint))], PUMP_PROGRAM_ID)[0])

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

class Tracker:
    def __init__(self):
        self.log = self._init_logger()
        self.session = None
        self.csv_lock = threading.Lock()
        self.active: Dict[str, SamplerTask] = {}
        self.migrated = asyncio.Event()

        # Check RugCheck status on startup
        if RUGCHECK_API_KEY:
            check_rugcheck_status()

        for fn, h in [(TRACKS_CSV, TRACK_HEADER), (EVENTS_CSV, EVENT_HEADER)]:
            if not os.path.exists(fn):
                with open(fn, "w", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=h).writeheader()

    def _init_logger(self):
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(f"logs/tracker_{time.strftime('%Y%m%d')}.log"),
                logging.StreamHandler(sys.stdout)
            ])
        return logging.getLogger("tracker")

    async def run(self):
        if not is_token_safe(ARGS.mint):
            self.log.info(f"❌ Skipping unsafe token {ARGS.mint} per RugCheck")
            return

        tasks = []
        try:
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=SSL_CTX)) as sess:
                self.session = sess
                self.log.info("RPC ready")

                mint_meta = {
                    "tokenAddress": ARGS.mint,
                    "name": ARGS.mint[:8],
                    "createdAt": now_iso(),
                    "trackable": True,
                }
                await self._start(mint_meta)

                if ARGS.debug_migrate:
                    tasks.append(asyncio.create_task(self._debug_migrate()))

                tasks += [
                    asyncio.create_task(self._listen_migration()),
                    asyncio.create_task(self._monitor()),
                    asyncio.create_task(self._heartbeat()),
                    asyncio.create_task(self._event_hub())
                ]

                await asyncio.gather(*tasks, return_exceptions=True)

                if ARGS.debug_launch:
                    self.log.info("🔧 Debug launching migrated tracker")
                    self.migrated.set()
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    async def _start(self, row):
        mint = row["tokenAddress"]
        launch = datetime.fromisoformat(row["createdAt"])
        age = (datetime.now(timezone.utc)-launch).total_seconds()
        if age > DURATION_SECS:
            return self.log.info(f"{mint} too old, skip")

        st = SamplerTask(mint, mint[:8], launch, curve_pda(mint), IntervalState(datetime.now(timezone.utc)))
        self.active[mint] = st
        asyncio.create_task(self._sampler(st))
        self.log.info(f"📡 {mint[:12]}… sampler started")

    async def _sampler(self, st: SamplerTask):
        try:
            while not self.migrated.is_set():
                age = (datetime.now(timezone.utc)-st.launch).total_seconds()
                if age / 3600 > ARGS.max_age_h:
                    self.log.info(f"⏰ Max age reached, stopping sampler for {st.name}")
                    break

                try:
                    row = await self._collect(st)
                    if row is None:
                        self.migrated.set()
                        break
                    row["migrated"] = self.migrated.is_set()
                    self._csv_write(TRACKS_CSV, row, TRACK_HEADER)
                except Exception as e:
                    self.log.error(f"{st.name} collect fail: {e}")

                await asyncio.sleep((ARGS.sampler_ms or 1000) / 1000)

            if self.migrated.is_set():
                try:
                    # Write final row
                    row = await self._collect(st)
                    if row:
                        row["migrated"] = True
                        self._csv_write(TRACKS_CSV, row, TRACK_HEADER)

                    # Verify migrated tracker exists
                    if not self._verify_migrated_tracker():
                        self.log.error(f"❌ Cannot launch migrated tracker for {st.mint}")
                        return

                    # Launch with output capture
                    self.log.info(f"🚀 Launching migrated tracker for {st.mint}")
                    result = subprocess.run([
                        sys.executable, "migrated_tracker.py",
                        "--mint", st.mint,
                        "--output-csv", TRACKS_CSV,
                        "--token-csv", f"{MIGRATED_TRACKS_DIR}/{st.mint}_tracks.csv"
                    ], capture_output=True, text=True)

                    if result.returncode != 0:
                        self.log.error(f"❌ Migrated tracker failed: {result.stderr}")
                    else:
                        self.log.info(f"✅ Migrated tracker launched for {st.mint}")
                        
                except Exception as e:
                    self.log.error(f"{st.name} migration handling failed: {e}")
        finally:
            self.active.pop(st.mint, None)
            self.log.info(f"✅ sampler done {st.name}")

    async def _collect(self, st):
        curve = await self._get_curve(st.curve)
        if not curve:
            return None

        v_sol, v_tok, r_sol = curve["v_sol"], curve["v_tok"], curve["r_sol"]
        sol_pool, raised = v_sol / 1e9, r_sol / 1e9
        prog = max(0, min(100, 100 - (curve["r_tok"] / INITIAL_RTOK_RES) * 100))

        flow  = raised - st.state.last_sol
        accel = flow   - st.state.last_R
        st.state.last_sol, st.state.last_R = raised, flow

        txs   = await self._recent_txs(st.curve)
        tm    = self._tx_metrics(txs)
        delta = tm["buy_vol"] - tm["sell_vol"]

        # 🔹 NEW — treat “progress 100 % AND pool drained” as migration
        if (
            prog >= 100.0                # full raise reached
            and sol_pool == 0.0          # bonding-curve emptied
            and not self.migrated.is_set()
        ):
            self.log.info(
                "🎓 progress=100 %% & pool=0 – assuming Raydium migration for %s",
                st.mint[:6],
            )
            self.migrated.set()
            return None                  # causes the sampler to break

        return dict(
            timestamp         = now_iso(),
            mint              = st.mint,
            sol_in_pool       = round(sol_pool, 6),
            sol_raised_total  = round(raised, 6),
            progress          = round(prog, 2),
            sol_flow          = round(flow, 6),
            sol_accel         = round(accel, 6),
            unique_buyers     = tm["unique"],
            top3_pct          = round(tm["top3_pct"], 2),
            lp_burn           = False,
            whale_flag        = tm["whale"],
            buy_sell_delta    = round(delta, 4),
            buy_pressure      = round(tm["buy_vol"], 5),
            sell_pressure     = round(tm["sell_vol"], 5),
            bot_like          = tm["bot"],
        )

    async def _get_curve(self, addr):
        payload = {"jsonrpc": "2.0", "id": 1, "method": "getAccountInfo",
                   "params": [addr, {"encoding": "base64"}]}
        val = (await self._rpc(payload)).get("result", {}).get("value")
        if not val or not val.get("data"):
            self.log.warning(f"🧨 PDA {addr} closed – migration assumed.")
            return None
        buf = base64.b64decode(val["data"][0])
        if len(buf) >= 81:
            v_tok, v_sol, r_tok, r_sol, *_ = struct.unpack("<QQQQQ?32x", buf[8:81])
            return dict(v_tok=v_tok, v_sol=v_sol, r_tok=r_tok, r_sol=r_sol)
        return None

    async def _recent_txs(self, curve):
        payload = {
            "jsonrpc": "2.0", "id": 1, "method": "getSignaturesForAddress",
            "params": [curve, {"limit": 200}]
        }
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
        signer_idx = next((i for i, a in enumerate(msg["accountKeys"]) if a.get("signer")), None)
        if signer_idx is None:
            return None
        sol = (post[signer_idx] - pre[signer_idx]) / 1e9
        if abs(sol) < MIN_VOL_SOL:
            return None
        return Tx(sig, msg["accountKeys"][signer_idx]["pubkey"], abs(sol), data["blockTime"], sol < 0)

    def _tx_metrics(self, txs: List[Tx]):
        buy, sell, buyers = 0, 0, {}
        for t in txs:
            if t.is_buy:
                buy += t.sol
                buyers[t.signer] = buyers.get(t.signer, 0) + t.sol
            else:
                sell += t.sol
        top3 = sum(sorted(buyers.values(), reverse=True)[:3]) / buy * 100 if buy else 0
        return dict(buy_vol=buy, sell_vol=sell, unique=len(buyers),
                    top3_pct=top3, whale=any(t.sol >= 1 for t in txs), bot=False)

    async def _event_hub(self): pass  # ✂️ You can paste the event hub if you need

    async def _listen_migration(self):
        if not self.active:
            return
        mint = next(iter(self.active.values())).mint
        async with self.session.ws_connect(WS, ssl=SSL_CTX) as ws:
            await ws.send_json({
                "jsonrpc": "2.0", "id": 1, "method": "blockSubscribe",
                "params": [{"mentionsAccountOrProgram": MIGRATION_ROUTER},
                           {"commitment": "confirmed", "encoding": "jsonParsed"}]
            })
            async for msg in ws:
                if msg.type is not aiohttp.WSMsgType.TEXT:
                    break
                j = msg.json()
                if j.get("method") != "blockNotification":
                    continue
                for tx in j["params"]["result"]["block"].get("transactions", []):
                    if tx.get("meta", {}).get("err"):
                        continue
                    for ix in tx["transaction"]["message"].get("instructions", []):
                        if ix.get("programId") == RAYDIUM_AMM and \
                           ix.get("parsed", {}).get("type") == INIT_OPCODE and \
                           len(ix.get("accounts", [])) > 8 and ix["accounts"][8] == mint:
                            self.migrated.set()
                            self.log.info(f"🎓 Raydium migration for {mint}")
                            return

    async def _debug_migrate(self):
        self.log.info("⚠️ Debug migration in 30s")
        await asyncio.sleep(30)
        self.migrated.set()
        self.log.info("🎓 Debug migration triggered")

    def _csv_write(self, fn, row, header):
        with self.csv_lock:
            with open(fn, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=header).writerow(row)

    async def _rpc(self, payload, retries=3):
        for a in range(retries):
            try:
                async with self.session.post(RPC, json=payload, timeout=8) as r:
                    j = await r.json()
                    if j.get("error"):
                        raise RuntimeError(j["error"].get("message", "RPC"))
                    return j
            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError):
                if a == retries-1:
                    raise
                await asyncio.sleep(0.4 * 2**a)

    async def _monitor(self):
        SAFETY_CHECK_INTERVAL = 30  # Check every 30 seconds
        
        while self.active and not self.migrated.is_set():
            try:
                # Check safety for all active tokens
                for mint, st in list(self.active.items()):
                    if not is_token_safe(mint):
                        self.log.warning(f"⚠️ Token {st.name} became unsafe, stopping tracker")
                        self.active.pop(mint)
                        st.migration_event.set()  # Trigger clean shutdown
            
                # Regular monitor update
                self.log.info(f"📊 Monitor | Active={len(self.active)} | Safety checks OK")
                await asyncio.sleep(SAFETY_CHECK_INTERVAL)
                
            except Exception as e:
                self.log.error(f"Monitor error: {e}")
                await asyncio.sleep(5)  # Brief pause on error
            
        self.log.info("🎬 All samplers finished")

    async def _heartbeat(self):
        while not self.migrated.is_set() and self.active:
            self.log.info(f"⏰ heartbeat | active={len(self.active)}")
            await asyncio.sleep(300)

    def _verify_migrated_tracker(self):
        tracker_path = os.path.join(os.path.dirname(__file__), "migrated_tracker.py")
        if not os.path.exists(tracker_path):
            self.log.error("❌ migrated_tracker.py not found!")
            return False
        return True

if __name__ == "__main__":
    try:
        asyncio.run(Tracker().run())
    except KeyboardInterrupt:
        print("\n⏹ stopped")
