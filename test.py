#!/usr/bin/env python3
"""
pumpfun_full_tracker.py

1) Discover Pump.fun launches (InitializeMint instructions under Pump.fun's program).
2) Detect when those tokens migrate to a Raydium or Pump.swap pool (pool creation).
3) For each migrated token, start a 45‐minute tracking session sampling every 10 seconds,
   computing metrics: progress to 85 SOL, flow, acceleration, unique buyers, top‐3 concentration,
   LP‐burn flag, token price, market cap (USD), whale flag, buy/sell delta, buy pressure,
   sell pressure, bot‐like flag.

All methods used are defined in this single script.
"""

import asyncio
import json
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Callable
import csv

import aiohttp
from solders.pubkey import Pubkey
from solders.signature import Signature
from solana.rpc.async_api import AsyncClient
from websockets import connect

# ─────────── Configuration ───────────

# Program IDs
PUMPFUN_PROGRAM_ID = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
RAYDIUM_AMM_ID = "9WnPzXuDpgc6PGaV7rdLbXP4Bh1eZVnR71W9DZPJgGGo"
# Replace with actual ID
PUMPSWAP_PROGRAM_ID = "PumpSwap111111111111111111111111111111111"

# RPC endpoints - Use a paid RPC service for production
RPC_WS_URL = "wss://api.mainnet-beta.solana.com/"
RPC_HTTP_URL = "https://api.mainnet-beta.solana.com"

# Alternative: Use a paid RPC service like Helius, QuickNode, or Alchemy
# RPC_HTTP_URL = "https://rpc.helius.xyz/?api-key=YOUR_API_KEY"
# RPC_WS_URL = "wss://rpc.helius.xyz/?api-key=YOUR_API_KEY"

# CoinGecko URL (Solana → USD)
COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd"

# Polling / tracking parameters
POLL_INTERVAL = 30    # Seconds between checking for new instructions
TRACK_INTERVAL = 60   # Increased to 60 seconds to reduce API calls
TRACK_DURATION_SECS = 45 * 60  # 45 minutes in seconds

# Rate limiting - Even more conservative settings
MAX_RPC_CALLS_PER_SECOND = 0.5  # Reduced to 1 call every 2 seconds
RPC_CALL_DELAY = 1.0 / MAX_RPC_CALLS_PER_SECOND

# Reduce data fetching
MAX_SIGNATURES_PER_CALL = 10  # Reduced from 100
MAX_TRANSACTIONS_PER_INTERVAL = 5  # Process fewer transactions

# State file for last‐seen timestamps
LAUNCH_STATE_FILE = Path("pumpfun_launch_state.json")
MIGRATE_STATE_FILE = Path("pumpfun_migrate_state.json")


# ─────────── Utility Functions ───────────

def iso_now() -> str:
    """Return current UTC time as ISO8601 string, e.g. '2025-06-05T18:00:00Z'."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_state(file: Path, default: Dict) -> Dict:
    """Load JSON state from a file; return default dict if missing/invalid."""
    if file.exists():
        try:
            return json.loads(file.read_text(encoding="utf-8"))
        except Exception:
            return default.copy()
    return default.copy()


def save_state(file: Path, data: Dict):
    """Save JSON state to a file (overwrites)."""
    file.write_text(json.dumps(data, indent=2), encoding="utf-8")


async def fetch_sol_price() -> float:
    """
    Fetch current SOL→USD price from CoinGecko.
    Returns a float (e.g. 33.21).
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(COINGECKO_URL) as resp:
            data = await resp.json()
            return float(data.get("solana", {}).get("usd", 0.0))


async def safe_rpc_call(client, method, *args, **kwargs):
    """
    Enhanced wrapper for RPC calls with better retry logic and rate limiting.
    """
    max_retries = 5  # Increased retries
    base_delay = 2.0  # Longer base delay
    
    for attempt in range(max_retries):
        try:
            # More aggressive rate limiting
            await asyncio.sleep(RPC_CALL_DELAY)
            
            # Make the RPC call
            if method == 'get_transaction':
                return await client.get_transaction(*args, **kwargs)
            elif method == 'get_token_account_balance':
                return await client.get_token_account_balance(*args, **kwargs)
            elif method == 'get_token_supply':
                return await client.get_token_supply(*args, **kwargs)
            elif method == 'get_signatures_for_address':
                return await client.get_signatures_for_address(*args, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method}")
                
        except Exception as e:
            error_msg = str(e)
            print(f"RPC error in {method} (attempt {attempt + 1}/{max_retries}): {error_msg}")
            
            if "429" in error_msg or "Too Many Requests" in error_msg:
                if attempt < max_retries - 1:
                    delay = base_delay * (3 ** attempt)  # More aggressive backoff
                    print(f"Rate limited, retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    print(f"Max retries exceeded for {method} due to rate limiting")
                    return None
            elif any(code in error_msg for code in ["502", "503", "504", "timeout"]):
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"Server error, retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    print(f"Max retries exceeded for {method} due to server errors")
                    return None
            else:
                print(f"Non-retryable error in {method}: {error_msg}")
                return None
    return None


# ─────────── Discover Pump.fun Launches ───────────

async def discover_pumpfun_launches(on_launch: Callable[[str, int], None]):
    """
    Subscribes to Solana logs mentioning Pump.fun's program ID.
    Whenever an InitializeMint instruction appears, extracts the new mint address
    and its blockTime, then calls on_launch(mint, timestamp).

    on_launch: callback(mint_address: str, launch_time_unix: int)
    """
    print(f"[{iso_now()}] Starting Pump.fun launch discovery...")
    async with connect(RPC_WS_URL) as ws:
        # 1) Subscribe to logs for Pump.fun
        await ws.send(json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "logsSubscribe",
            "params": [
                {"mentions": [PUMPFUN_PROGRAM_ID]},
                {"commitment": "confirmed"}
            ]
        }))
        # Wait for subscription confirmation
        sub_response = await ws.recv()
        print(f"Subscription response: {sub_response}")

        http_client = AsyncClient(RPC_HTTP_URL)

        try:
            print(f"[{iso_now()}] Listening for Pump.fun transactions...")
            while True:
                message = await ws.recv()
                data = json.loads(message)
                
                # Only consider logsNotification messages
                if data.get("method") != "logsNotification":
                    print(f"Received non-log message: {data.get('method', 'unknown')}")
                    continue

                # The actual payload is under data["params"]["result"]["value"]
                params = data["params"]["result"]
                value = params["value"]
                signature = value.get("signature")
                block_time = value.get("blockTime")

                print(f"[{iso_now()}] Processing log for signature: {signature}")

                # If any log line contains "Instruction: InitializeMint", it's a new mint
                logs = value.get("logs", [])
                has_init_mint = any("Instruction: InitializeMint" in line for line in logs)
                
                if has_init_mint:
                    print(f"[{iso_now()}] Found InitializeMint in transaction {signature}")
                    
                    # Fetch full transaction in jsonParsed to parse instructions
                    tx_resp = await safe_rpc_call(
                        http_client, 
                        'get_transaction',
                        Signature.from_string(signature), 
                        encoding="jsonParsed", 
                        commitment="confirmed",
                        max_supported_transaction_version=0
                    )
                    
                    if not tx_resp or not tx_resp.value:
                        print(f"Failed to fetch transaction details for {signature}")
                        continue

                    tx_result = tx_resp.value
                    
                    # Handle the transaction structure properly
                    if hasattr(tx_result, 'transaction'):
                        transaction = tx_result.transaction
                        if hasattr(transaction, 'message'):
                            message_obj = transaction.message
                        else:
                            print(f"Transaction {signature} has no message attribute")
                            continue
                    else:
                        print(f"Transaction {signature} has unexpected structure")
                        continue

                    # Look through instructions for SPL Token initializeMint
                    if hasattr(message_obj, 'instructions'):
                        for instr in message_obj.instructions:
                            # Check if this is an SPL Token instruction
                            if hasattr(instr, 'program_id'):
                                program_id = str(instr.program_id)
                                if program_id == "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA":
                                    # Check if it's a parsed instruction with initializeMint type
                                    if hasattr(instr, 'parsed') and isinstance(instr.parsed, dict):
                                        parsed_info = instr.parsed
                                        if parsed_info.get("type") == "initializeMint":
                                            mint_addr = parsed_info.get("info", {}).get("mint")
                                            if mint_addr and block_time is not None:
                                                print(f"[{iso_now()}] Found new mint: {mint_addr}")
                                                on_launch(str(mint_addr), block_time)
                                            break
                    else:
                        print(f"Transaction {signature} message has no instructions")
                else:
                    # Print some logs to see what we're receiving
                    log_preview = logs[:3] if logs else []
                    print(f"[{iso_now()}] Non-InitializeMint transaction, sample logs: {log_preview}")

        except asyncio.CancelledError:
            print(f"[{iso_now()}] Launch discovery cancelled")
            pass
        except Exception as e:
            print(f"Error in launch discovery: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await http_client.close()


# ─────────── Detect Migration to Raydium or Pump.swap ───────────

async def detect_migration(
    launched_tokens: Dict[str, int],
    migrated_callback: Callable[[str, str, int], None]
):
    """
    Subscribes to logs for Raydium AMM and Pump.swap program IDs.
    When a pool‐creation instruction mentions a launched token, calls:
       migrated_callback(mint, pool_address, migrate_time)

    launched_tokens: dict mapping mint_address -> launch_time_unix
    migrated_callback: callback(mint_address: str, pool_address: str, migrate_time: int)
    """
    print(f"[{iso_now()}] Starting migration detection...")
    async with connect(RPC_WS_URL) as ws:
        # 1) Subscribe to logs for both Raydium and Pump.swap
        await ws.send(json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "logsSubscribe",
            "params": [
                {"mentions": [RAYDIUM_AMM_ID, PUMPSWAP_PROGRAM_ID]},
                {"commitment": "confirmed"}
            ]
        }))
        await ws.recv()  # subscription ack

        http_client = AsyncClient(RPC_HTTP_URL)

        try:
            while True:
                message = await ws.recv()
                data = json.loads(message)
                if data.get("method") != "logsNotification":
                    continue

                params = data["params"]["result"]
                value = params["value"]
                signature = value.get("signature")
                block_time = value.get("blockTime")
                logs = value.get("logs", [])

                # If any log line contains "InitPool" or "createPool", we suspect a pool was made
                if any(
                    ("Instruction: InitPool" in line) or (
                        "Instruction: createPool" in line)
                    for line in logs
                ):
                    tx_resp = await safe_rpc_call(
                        http_client, 
                        'get_transaction',
                        Signature.from_string(signature), 
                        encoding="jsonParsed", 
                        commitment="confirmed",
                        max_supported_transaction_version=0
                    )
                    
                    if not tx_resp or not tx_resp.value:
                        continue

                    tx_result = tx_resp.value
                    
                    # Handle the transaction structure properly
                    if hasattr(tx_result, 'transaction'):
                        transaction = tx_result.transaction
                        if hasattr(transaction, 'message'):
                            message_obj = transaction.message
                        else:
                            continue
                    else:
                        continue

                    # Gather all accounts that match a launched mint
                    involved_mints = set()
                    if hasattr(message_obj, 'instructions'):
                        for instr in message_obj.instructions:
                            if hasattr(instr, 'accounts'):
                                for acc_pub in instr.accounts:
                                    acc_str = str(acc_pub)
                                    if acc_str in launched_tokens:
                                        involved_mints.add(acc_str)

                    for mint in involved_mints:
                        pool_address = None
                        # Heuristic: find the first non‐mint account under a Raydium/Pump.swap instruction
                        if hasattr(message_obj, 'instructions'):
                            for instr in message_obj.instructions:
                                if hasattr(instr, 'program_id'):
                                    prog = str(instr.program_id)
                                    if prog in (RAYDIUM_AMM_ID, PUMPSWAP_PROGRAM_ID):
                                        if hasattr(instr, 'accounts'):
                                            for acc_pub in instr.accounts:
                                                acc_str = str(acc_pub)
                                                if acc_str != mint:
                                                    pool_address = acc_str
                                                    break
                                        if pool_address:
                                            break

                        if pool_address and block_time is not None:
                            migrated_callback(mint, pool_address, block_time)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Error in migration detection: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await http_client.close()


# ─────────── Token‐By‐Token Tracking Session ───────────

class TokenTracker:
    """
    Tracks a single token AFTER migration (pool creation). Samples every 10s for 45min,
    computing all required metrics each interval.
    """

    def __init__(self, mint: str, pool: str, launch_time: int, migrate_time: int, client: AsyncClient):
        self.mint = mint
        self.pool = pool
        self.launch_time = launch_time
        self.migrate_time = migrate_time
        self.client = client  # shared AsyncClient for RPC calls

        # History for SOL balances (for flow & acceleration)
        self.sol_history = deque(maxlen=2)   # keep [prev, current]
        self.flow_history = deque(maxlen=2)  # keep [prev_flow, current_flow]

        # For tracking buys in last 20s (whale & bot flags)
        # buys_last_20s: deque of (user_pubkey, amount, timestamp)
        self.buys_last_20s: deque = deque()
        # Per‐user deque of (timestamp, amount) for bot detection
        self.buy_events_last_20s: Dict[str,
                                       deque] = defaultdict(lambda: deque())

        # Flag for LP burns this window
        self.lp_burned = False

        # Storage for computed metrics over time
        self.metrics: list = []

        # Create CSV file for this token
        self.csv_file = Path(f"token_metrics_{mint[:8]}.csv")
        self.csv_headers = [
            "timestamp", "progress_85_SOL", "interval_flow", "acceleration",
            "unique_buyers", "top3_concentration", "lp_burn_flag", 
            "token_price_SOL", "market_cap_USD", "whale_flag", 
            "buy_sell_delta", "buy_pressure", "sell_pressure", "bot_flag"
        ]
        
        # Write CSV headers
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_headers)
            writer.writeheader()

    async def get_pool_sol(self) -> float:
        """
        Fetch the SOL (wrapped SOL) vault balance for this pool.
        In a real scenario, you'd derive the wSOL vault PDA from the pool address.
        Here we'll assume `self.pool` itself is the wSOL vault for simplicity.
        """
        try:
            resp = await safe_rpc_call(self.client, 'get_token_account_balance', Pubkey(self.pool))
            if resp and resp.get("result"):
                val = resp["result"]["value"]["uiAmount"]
                return float(val) if val is not None else 0.0
        except Exception as e:
            print(f"Error getting pool SOL: {e}")
        return 0.0

    async def get_pool_token(self) -> float:
        """
        Fetch the SPL token vault balance for this pool.
        In reality, derive the pool's token vault PDA; here we reuse `self.pool`.
        """
        try:
            resp = await safe_rpc_call(self.client, 'get_token_account_balance', Pubkey(self.pool))
            if resp and resp.get("result"):
                val = resp["result"]["value"]["uiAmount"]
                return float(val) if val is not None else 0.0
        except Exception as e:
            print(f"Error getting pool token: {e}")
        return 0.0

    async def get_total_supply(self) -> float:
        """
        Fetch the total supply of the token mint.
        """
        try:
            resp = await safe_rpc_call(self.client, 'get_token_supply', Pubkey(self.mint))
            if resp and resp.get("result"):
                val = resp["result"]["value"]["uiAmount"]
                return float(val) if val is not None else 0.0
        except Exception as e:
            print(f"Error getting total supply: {e}")
        return 0.0

    async def sample_interval(self):
        """
        Enhanced sample_interval with better rate limit handling.
        """
        # 1) Fetch current pool SOL and token balances with delays
        sol_now = await self.get_pool_sol()
        await asyncio.sleep(0.5)  # Small delay between calls
        
        token_now = await self.get_pool_token()
        await asyncio.sleep(0.5)
        
        supply_now = await self.get_total_supply()
        await asyncio.sleep(0.5)

        # 2) Compute price and append to history
        price_SOL = sol_now / token_now if token_now > 0 else 0.0

        # 3) Append SOL balance to history & compute flow/acceleration
        if len(self.sol_history) == 0:
            self.sol_history.append(sol_now)
            self.flow_history.append(0.0)
            flow = 0.0
            accel = 0.0
        else:
            prev_sol = self.sol_history[-1]
            flow = sol_now - prev_sol
            self.flow_history.append(flow)
            if len(self.flow_history) < 2:
                accel = 0.0
            else:
                accel = self.flow_history[-1] - self.flow_history[-2]
            self.sol_history.append(sol_now)

        # 4) Fetch recent swap & LP logs with reduced limits
        window_start_ts = int(time.time()) - (TRACK_INTERVAL + 1)
        
        sigs_resp = await safe_rpc_call(
            self.client,
            'get_signatures_for_address',
            Pubkey(self.pool),
            limit=MAX_SIGNATURES_PER_CALL,  # Much smaller limit
        )
        
        if not sigs_resp or not sigs_resp.value:
            print("No signatures found or rate limited - using empty metrics")
            # Still compute basic metrics with zero values
            # Initialize interval aggregates
            buys_volume = defaultdict(float)
            sells_volume = defaultdict(float)
            unique_buyers = set()
            total_buy = 0.0
            total_sell = 0.0
            lp_burn_flag = False

            now_ts = time.time()

            # 6) Unique buyers U
            U = len(unique_buyers)

            # 7) Top‐3 concentration C
            sorted_buys = sorted(buys_volume.values(), reverse=True)
            top3 = sum(sorted_buys[:3]) if sorted_buys else 0.0
            C = top3 / (total_buy if total_buy > 0 else 1.0)

            # 8) Whale flag: any single wallet bought ≥ 1 SOL in last 20 s
            # Purge old events from buys_last_20s and buy_events_last_20s
            while self.buys_last_20s and now_ts - self.buys_last_20s[0][2] > 20:
                user_old, amt_old, ts_old = self.buys_last_20s.popleft()
                user_queue = self.buy_events_last_20s[user_old]
                # Remove matching event from user_queue
                while user_queue and user_queue[0][0] == ts_old:
                    user_queue.popleft()
            whale_flag = any(
                any(amt >= 1.0 for (_, amt) in events) for events in self.buy_events_last_20s.values()
            )

            # 9) Bot‐like flag: any wallet made ≥ 4 buys of <0.1 SOL within 0.2 s
            bot_flag = False
            for user, events in self.buy_events_last_20s.items():
                # events is deque of (ts, amt). Check sliding window of 0.2s
                times = [ts for (ts, amt) in events if amt < 0.1]
                for i in range(len(times)):
                    count = 1
                    base_ts = times[i]
                    for j in range(i + 1, len(times)):
                        if times[j] - base_ts <= 0.2:
                            count += 1
                        else:
                            break
                    if count >= 4:
                        bot_flag = True
                        break
                if bot_flag:
                    break

            # 10) Buy/sell delta & pressures
            delta = total_buy - total_sell
            buy_pressure = total_buy
            sell_pressure = total_sell

            # 11) Progress toward 85 SOL P
            P = min(sol_now / 85.0, 1.0) if sol_now >= 0 else 0.0

            # 12) Market cap (USD)
            sol_price_usd = await fetch_sol_price()
            price_USD = price_SOL * sol_price_usd
            market_cap_usd = price_USD * supply_now

            # 13) Compile metrics for this window
            metrics = {
                "timestamp": now_ts,
                "progress_85_SOL": P,
                "interval_flow": flow,
                "acceleration": accel,
                "unique_buyers": U,
                "top3_concentration": C,
                "lp_burn_flag": lp_burn_flag,
                "token_price_SOL": price_SOL,
                "market_cap_USD": market_cap_usd,
                "whale_flag": whale_flag,
                "buy_sell_delta": delta,
                "buy_pressure": buy_pressure,
                "sell_pressure": sell_pressure,
                "bot_flag": bot_flag,
            }
            
            # Store to CSV
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_headers)
                writer.writerow(metrics)
            
            # Also print to console
            print(json.dumps(metrics, indent=2))

            return

        sigs = sigs_resp.value
        recent_sigs = [s.signature for s in sigs if (s.block_time or 0) >= window_start_ts]

        # Process even fewer transactions
        recent_sigs = recent_sigs[:MAX_TRANSACTIONS_PER_INTERVAL]

        # Initialize interval aggregates
        buys_volume = defaultdict(float)
        sells_volume = defaultdict(float)
        unique_buyers = set()
        total_buy = 0.0
        total_sell = 0.0
        lp_burn_flag = False

        now_ts = time.time()

        # 5) Parse each recent transaction
        for sig in recent_sigs:
            tx_resp = await safe_rpc_call(
                self.client,
                'get_transaction',
                Signature.from_string(str(sig)), 
                encoding="jsonParsed", 
                commitment="confirmed",
                max_supported_transaction_version=0
            )
            
            if not tx_resp or not tx_resp.value:
                continue

            tx_result = tx_resp.value
            
            # Handle the transaction structure properly
            if hasattr(tx_result, 'transaction'):
                transaction = tx_result.transaction
                if hasattr(transaction, 'message'):
                    message_obj = transaction.message
                else:
                    continue
            else:
                continue

            if hasattr(message_obj, 'instructions') and hasattr(message_obj, 'account_keys'):
                for instr in message_obj.instructions:
                    if hasattr(instr, 'program_id'):
                        prog = str(instr.program_id)
                        # assume first key is the user
                        user = str(message_obj.account_keys[0]) if message_obj.account_keys else ""

                        # 5a) Detect buys/sells in Raydium or Pump.swap
                        if prog in (RAYDIUM_AMM_ID, PUMPSWAP_PROGRAM_ID):
                            if hasattr(instr, 'parsed') and isinstance(instr.parsed, dict):
                                parsed = instr.parsed
                                instr_type = parsed.get("type", "")
                                info = parsed.get("info", {})

                                # Typical swap instructions have 'amountIn' and 'amountOut'
                                amt_in = float(info.get("amountIn", 0))
                                amt_out = float(info.get("amountOut", 0))

                                # If user pays SOL (amountIn>0) and receives tokens (amountOut>0), count as buy
                                if instr_type in ("swap", "swapExactIn", "swapExactOut") and amt_in > 0 and amt_out > 0:
                                    buys_volume[user] += amt_in
                                    total_buy += amt_in
                                    unique_buyers.add(user)
                                    # Record buy in last‐20s structures
                                    self.buys_last_20s.append((user, amt_in, now_ts))
                                    self.buy_events_last_20s[user].append((now_ts, amt_in))

                                # Otherwise, if user receives SOL (amtIn>0 but amountOut refers to SOL), treat as sell
                                elif instr_type in ("swap", "swapExactIn", "swapExactOut") and amt_in > 0 and amt_out == 0:
                                    sells_volume[user] += amt_in
                                    total_sell += amt_in

                                # 5b) Detect LP burns (removeLiquidity, withdraw, etc.)
                                if instr_type in ("removeLiquidity", "withdraw", "burnLiquidity"):
                                    lp_burn_flag = True

                        # 5c) SPL Token burn by pool (rare but track if any)
                        if prog == "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA" and hasattr(instr, 'parsed') and isinstance(instr.parsed, dict) and instr.parsed.get("type") == "burn":
                            lp_burn_flag = True

        # 6) Unique buyers U
        U = len(unique_buyers)

        # 7) Top‐3 concentration C
        sorted_buys = sorted(buys_volume.values(), reverse=True)
        top3 = sum(sorted_buys[:3]) if sorted_buys else 0.0
        C = top3 / (total_buy if total_buy > 0 else 1.0)

        # 8) Whale flag: any single wallet bought ≥ 1 SOL in last 20 s
        # Purge old events from buys_last_20s and buy_events_last_20s
        while self.buys_last_20s and now_ts - self.buys_last_20s[0][2] > 20:
            user_old, amt_old, ts_old = self.buys_last_20s.popleft()
            user_queue = self.buy_events_last_20s[user_old]
            # Remove matching event from user_queue
            while user_queue and user_queue[0][0] == ts_old:
                user_queue.popleft()
        whale_flag = any(
            any(amt >= 1.0 for (_, amt) in events) for events in self.buy_events_last_20s.values()
        )

        # 9) Bot‐like flag: any wallet made ≥ 4 buys of <0.1 SOL within 0.2 s
        bot_flag = False
        for user, events in self.buy_events_last_20s.items():
            # events is deque of (ts, amt). Check sliding window of 0.2s
            times = [ts for (ts, amt) in events if amt < 0.1]
            for i in range(len(times)):
                count = 1
                base_ts = times[i]
                for j in range(i + 1, len(times)):
                    if times[j] - base_ts <= 0.2:
                        count += 1
                    else:
                        break
                if count >= 4:
                    bot_flag = True
                    break
            if bot_flag:
                break

        # 10) Buy/sell delta & pressures
        delta = total_buy - total_sell
        buy_pressure = total_buy
        sell_pressure = total_sell

        # 11) Progress toward 85 SOL P
        P = min(sol_now / 85.0, 1.0) if sol_now >= 0 else 0.0

        # 12) Market cap (USD)
        sol_price_usd = await fetch_sol_price()
        price_USD = price_SOL * sol_price_usd
        market_cap_usd = price_USD * supply_now

        # 13) Compile metrics for this window
        metrics = {
            "timestamp": now_ts,
            "progress_85_SOL": P,
            "interval_flow": flow,
            "acceleration": accel,
            "unique_buyers": U,
            "top3_concentration": C,
            "lp_burn_flag": lp_burn_flag,
            "token_price_SOL": price_SOL,
            "market_cap_USD": market_cap_usd,
            "whale_flag": whale_flag,
            "buy_sell_delta": delta,
            "buy_pressure": buy_pressure,
            "sell_pressure": sell_pressure,
            "bot_flag": bot_flag,
        }
        
        # Store to CSV
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_headers)
            writer.writerow(metrics)
        
        # Also print to console
        print(json.dumps(metrics, indent=2))

    async def track(self):
        """
        Starts the 45-minute, 10-second‐interval tracking for this token/pool.
        """
        print(f"[{iso_now()}] Starting tracking for {self.mint} (pool {self.pool})")
        # Initialize sol_history with the current SOL balance
        sol0 = await self.get_pool_sol()
        self.sol_history.append(sol0)
        self.flow_history.append(0.0)

        # For the first window, flow & accel are zero
        start_time = time.time()
        end_time = start_time + TRACK_DURATION_SECS

        while time.time() < end_time:
            await self.sample_interval()
            # Sleep until next 10‐second tick
            await asyncio.sleep(TRACK_INTERVAL)

        print(f"[{iso_now()}] Finished tracking for {self.mint}")


# ─────────── Main Runner ───────────

async def main():
    # 1) Shared RPC client for all trackers
    client = AsyncClient(RPC_HTTP_URL)

    # 2) In‐memory state for launches and migrations
    # Load previous state (optional); for simplicity, start fresh each run
    launched_tokens: Dict[str, int] = {}   # mint -> launch_time_unix
    migrating_tokens: Dict[str, int] = {}  # mint -> migrate_time_unix

    # 3) Callback when a new mint is launched via Pump.fun
    def on_launch(mint: str, ts: int):
        if mint not in launched_tokens:
            launched_tokens[mint] = ts
            print(f"[{iso_now()}] [LAUNCH] {mint} at {ts}")

    # 4) Callback when a launched token is detected migrating to a pool
    def on_migrate(mint: str, pool: str, ts: int):
        # Only track once per mint
        if mint in launched_tokens and mint not in migrating_tokens:
            migrating_tokens[mint] = ts
            print(f"[{iso_now()}] [MIGRATE] {mint} → pool {pool} at {ts}")
            # Create and start a TokenTracker in the background
            tracker = TokenTracker(mint=mint, pool=pool,
                                   launch_time=launched_tokens[mint],
                                   migrate_time=ts,
                                   client=client)
            asyncio.create_task(tracker.track())

    # 5) Run both discovery coroutines with retry logic
    while True:
        try:
            await asyncio.gather(
                discover_pumpfun_launches_with_retry(on_launch),
                detect_migration_with_retry(launched_tokens, on_migrate),
            )
        except Exception as e:
            print(f"[{iso_now()}] Main loop error: {e}")
            print(f"[{iso_now()}] Restarting in 10 seconds...")
            await asyncio.sleep(10)


async def discover_pumpfun_launches_with_retry(on_launch: Callable[[str, int], None]):
    """
    Wrapper for discover_pumpfun_launches with retry logic for connection drops.
    """
    max_retries = 5
    retry_delay = 5.0
    
    for attempt in range(max_retries):
        try:
            print(f"[{iso_now()}] Starting launch discovery (attempt {attempt + 1}/{max_retries})")
            await discover_pumpfun_launches(on_launch)
        except Exception as e:
            print(f"[{iso_now()}] Launch discovery failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"[{iso_now()}] Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 1.5  # Exponential backoff
            else:
                print(f"[{iso_now()}] Max retries exceeded for launch discovery")
                raise

async def detect_migration_with_retry(launched_tokens: Dict[str, int], migrated_callback: Callable[[str, str, int], None]):
    """
    Wrapper for detect_migration with retry logic for connection drops.
    """
    max_retries = 5
    retry_delay = 5.0
    
    for attempt in range(max_retries):
        try:
            print(f"[{iso_now()}] Starting migration detection (attempt {attempt + 1}/{max_retries})")
            await detect_migration(launched_tokens, migrated_callback)
        except Exception as e:
            print(f"[{iso_now()}] Migration detection failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"[{iso_now()}] Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 1.5  # Exponential backoff
            else:
                print(f"[{iso_now()}] Max retries exceeded for migration detection")
                raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n[{iso_now()}] Exiting on user interrupt.")
