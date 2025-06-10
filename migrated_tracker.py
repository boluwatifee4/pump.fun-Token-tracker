#!/usr/bin/env python3
"""
migrated_tracker.py – Raydium-stage tracker

• Samples the two Raydium vaults every --interval seconds
• Correctly scales balances with token decimals
• Scrapes recent swap txs to fill buy/sell metrics
• Appends to a global CSV and (optionally) a per-token CSV
"""

import asyncio, aiohttp, csv, os, sys, time, logging, threading, base64, struct
from datetime import datetime, timezone
from typing import Optional, Dict, List, Set
from solana.publickey import PublicKey
import argparse

# --- CLI ----------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--mint",        required=True,               help="Token mint")
ap.add_argument("--output-csv",  default="all_token_tracks.csv")
ap.add_argument("--token-csv",   help="Optional per-token file")
ap.add_argument("--interval",    type=float, default=5.0,     help="Sampling seconds")
ARGS = ap.parse_args()

# --- Environment --------------------------------------------------------------
HELIUS_KEY = os.getenv("HELIUS_KEY")
if not HELIUS_KEY:
    print("❌ HELIUS_KEY env var not set");  sys.exit(1)

RPC = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_KEY}"

RAYDIUM_AMM = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"   # AMM v4
MIN_VOL_SOL = 0.00005
TRACK_HEADER = [
    "timestamp","mint","sol_in_pool","sol_raised_total","progress",
    "sol_flow","sol_accel","unique_buyers","top3_pct","lp_burn",
    "whale_flag","buy_sell_delta","buy_pressure","sell_pressure","bot_like",
    "migrated"
]

# --- Logging ------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(f"logs/migrated_{time.strftime('%Y%m%d')}.log"),
        logging.StreamHandler(sys.stdout)
    ])
log = logging.getLogger("migrated-tracker")

MIGRATED_TRACKS_DIR = "migrated_tracks"
os.makedirs(MIGRATED_TRACKS_DIR, exist_ok=True)

# --- Helpers ------------------------------------------------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

_csv_lock = threading.Lock()
# Update csv_write function to validate file paths
def csv_write(fn: str, row: Dict):
    if not fn:  # Add validation for empty path
        log.error("Empty file path provided to csv_write")
        return
        
    with _csv_lock:
        try:
            # Ensure parent directory exists
            dirpath = os.path.dirname(fn)
            if dirpath:  # Only create dir if path has a directory component
                os.makedirs(dirpath, exist_ok=True)
            
            first = not os.path.exists(fn)
            with open(fn, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=TRACK_HEADER)
                if first: 
                    w.writeheader()
                w.writerow(row)
        except Exception as e:
            log.error(f"Failed to write to {fn}: {e}")

async def _rpc(session, payload, retries=3):
    for a in range(retries):
        try:
            async with session.post(RPC, json=payload, timeout=8) as r:
                j = await r.json()
                if j.get("error"): raise RuntimeError(j["error"]["message"])
                return j["result"]
        except Exception as e:
            if a == retries-1: raise
            await asyncio.sleep(0.4*2**a)

# --- Chain queries ------------------------------------------------------------
async def get_mint_decimals(session, mint:str) -> int:
    res = await _rpc(session, {
        "jsonrpc":"2.0","id":1,"method":"getAccountInfo",
        "params":[mint,{"encoding":"base64"}]})
    data = base64.b64decode(res["value"]["data"][0])
    decimals = data[44]              # SPL-Token mint layout: u8 at offset 44
    return decimals

async def resolve_vaults(session, mint: str):
    """
    Return (WSOL-vault, token-vault).
    Scans the top-10 largest accounts and picks the two whose
    parsed.data.parsed.info.mint equals either SOL-wrapped or the token mint.
    """
    # 1. Fetch the largest token accounts for this mint
    result = await _rpc(session, {
        "jsonrpc": "2.0", "id": 1,
        "method": "getTokenLargestAccounts",
        "params": [mint]
    })

    # 2. Look for the two vaults by checking the 'mint' field in parsed.info
    SOL_MINT = "So11111111111111111111111111111111111111112"
    vaults: List[tuple[str, str]] = []  # list of (address, mint)

    for acct in result["value"][:10]:  # scan a few in case of dust holders
        addr = acct["address"]
        info = await _rpc(session, {
            "jsonrpc": "2.0", "id": 1,
            "method": "getAccountInfo",
            "params": [addr, {"encoding": "jsonParsed"}]
        })

        parsed = (
            info.get("value", {})
                .get("data", {})
                .get("parsed", {})
                .get("info", {})
        )
        acct_mint = parsed.get("mint")
        # match exactly SOL-wrapped vault or the token vault
        if acct_mint in (SOL_MINT, mint):
            vaults.append((addr, acct_mint))
            if len(vaults) == 2:
                break

    if len(vaults) != 2:
        raise RuntimeError(f"Could not resolve both Raydium vaults for {mint}")

    # 3. Sort so that vault_sol is the SOL-wrapped one, vault_tok is the token vault
    if vaults[0][1] == SOL_MINT:
        vault_sol, vault_tok = vaults[0][0], vaults[1][0]
    else:
        vault_sol, vault_tok = vaults[1][0], vaults[0][0]

    return vault_sol, vault_tok

async def get_token_balance(session, account:str) -> int:
    res = await _rpc(session,{
        "jsonrpc":"2.0","id":1,"method":"getTokenAccountBalance","params":[account]})
    return int(res["value"]["amount"])

async def recent_sigs(session, addr:str, limit=100):
    res = await _rpc(session,{
        "jsonrpc":"2.0","id":1,"method":"getSignaturesForAddress",
         "params":[addr,{"limit":limit}]})
    return [s["signature"] for s in res]

async def fetch_tx(session, sig:str):
    res = await _rpc(session,{
        "jsonrpc":"2.0","id":1,"method":"getTransaction",
         "params":[sig,{"encoding":"jsonParsed","maxSupportedTransactionVersion":0}]})
    return res

# --- Metrics accumulation -----------------------------------------------------
class SwapTracker:
    def __init__(self):
        self.seen: Set[str] = set()

    async def collect(self, session, vault_sol: str) -> Dict:
        sigs = await recent_sigs(session, vault_sol, 200)
        new_sigs = [s for s in sigs if s not in self.seen]
        if not new_sigs:
            return dict(buy=0, sell=0, buyers={})
        self.seen.update(new_sigs)

        buy = sell = 0.0
        buyers: Dict[str, float] = {}

        for sig in new_sigs:
            tx = await fetch_tx(session, sig)
            if not tx or tx.get("meta", {}).get("err"):
                continue

            keys = tx["transaction"]["message"]["accountKeys"]
            pre  = tx["meta"].get("preTokenBalances", [])
            post = tx["meta"].get("postTokenBalances", [])

            # 1) find vault index in accountKeys
            vault_idx = next(
                (i for i, acct in enumerate(keys) if acct.get("pubkey") == vault_sol),
                None
            )
            if vault_idx is None:
                continue

            # 2) match that index to a pre/post balance entry
            delta = None
            for p, q in zip(pre, post):
                if p.get("accountIndex") == vault_idx:
                    amt_pre  = int(p["uiTokenAmount"]["amount"])
                    amt_post = int(q["uiTokenAmount"]["amount"])
                    delta    = amt_post - amt_pre
                    break
            if delta is None:
                continue

            # 3) compute SOL delta and ignore dust
            sol = abs(delta) / 1e9
            if sol < MIN_VOL_SOL:
                continue

            # 4) find the signer
            signer = next((a.get("pubkey") for a in keys if a.get("signer")), None)
            if not signer:
                continue

            # 5) attribute buy vs. sell
            if delta > 0:
                buy += sol      # vault gained SOL → user sold token
            else:
                sell += sol     # vault lost SOL  → user bought token

            buyers[signer] = buyers.get(signer, 0.0) + sol

        return dict(buy=buy, sell=sell, buyers=buyers)

# --- Main loop ----------------------------------------------------------------
async def main():
    async with aiohttp.ClientSession() as session:
        # 1. discovery
        vault_sol, vault_tok = await resolve_vaults(session, ARGS.mint)
        dec_sol  = await get_mint_decimals(session, "So11111111111111111111111111111111111111112")
        dec_tok  = await get_mint_decimals(session, ARGS.mint)
        scale_sol, scale_tok = 10**dec_sol, 10**dec_tok
        log.info(f"vaults resolved | SOL-vault={vault_sol[:6]}…  TOK-vault={vault_tok[:6]}…")

        last_raised = last_flow = 0.0
        swap_tracker = SwapTracker()

        while True:
            try:
                bal_sol = await get_token_balance(session, vault_sol)
                bal_tok = await get_token_balance(session, vault_tok)
                sol_pool = bal_sol / scale_sol
                raised   = sol_pool                           # post-Pump TVL ≈ pool SOL

                flow  = raised - last_raised
                accel = flow   - last_flow
                last_raised, last_flow = raised, flow

                swap   = await swap_tracker.collect(session, vault_sol)
                buyers = swap["buyers"]
                top3   = sum(sorted(buyers.values(), reverse=True)[:3])
                top3_pct = top3 / swap["buy"] * 100 if swap["buy"] else 0

                row = dict(
                    timestamp      = now_iso(),
                    mint           = ARGS.mint,
                    sol_in_pool    = round(sol_pool, 6),
                    sol_raised_total = round(raised, 6),
                    progress       = 100.0,
                    sol_flow       = round(flow, 6),
                    sol_accel      = round(accel, 6),
                    unique_buyers  = len(buyers),
                    top3_pct       = round(top3_pct, 2),
                    lp_burn        = False,
                    whale_flag     = any(v >= 1 for v in buyers.values()),
                    buy_sell_delta = round(swap["buy"] - swap["sell"], 4),
                    buy_pressure   = round(swap["buy"], 5),
                    sell_pressure  = round(swap["sell"], 5),
                    bot_like       = False,          # placeholder
                    migrated       = True
                )

                # In main loop, add validation before writing
                if ARGS.output_csv:
                    csv_write(ARGS.output_csv, row)
                if ARGS.token_csv:
                    token_path = os.path.join(MIGRATED_TRACKS_DIR, f"{ARGS.mint}_tracks.csv")
                    csv_write(token_path, row)

                await asyncio.sleep(ARGS.interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"loop error: {e}")
                await asyncio.sleep(5)

# --- Entrypoint ---------------------------------------------------------------
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("🛑 stopped by user")
