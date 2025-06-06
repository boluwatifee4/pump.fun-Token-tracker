# 🚀 Pump.fun Token Tracker

A comprehensive Solana token tracking system for pump.fun tokens that monitors price movements, trading activity, and bonding curve progress in real-time.

## 📁 Project Structure

```
pump_tracker/
├── 📄 Core Scripts
│   ├── pf_tf.py              # Token fetcher (Moralis API)
│   ├── get_fresh_tokens.py   # Alternative token fetcher (RPC-based)
│   └── pf_tracker.py         # Main tracking engine
├── 🔧 Utilities
│   ├── debug_transactions.py # Transaction debugging tool
│   └── test_sources.py       # API testing utility
├── 📊 Data Files
│   ├── pump_tokens.csv       # Current tokens to track
│   ├── backlog_pump_tokens.csv # Historical token archive
│   └── requirements.txt      # Python dependencies
├── 📁 Output Directories
│   ├── logs/                 # System logs
│   └── tracks/               # Individual token tracking data
└── README.md                 # This file
```

## 🎯 What This System Does

### **Core Functionality:**

- **Discovers** newly launched pump.fun tokens
- **Tracks** 16 key metrics every 10 seconds for 45 minutes
- **Monitors** bonding curve progress, whale activity, bot detection
- **Archives** all data in CSV format for analysis

### **Key Metrics Tracked:**

- Progress (% towards migration)
- Buy/Sell volumes and pressure
- Unique buyer count
- Top 3 concentration
- Whale flags (>1 SOL trades)
- Bot-like activity detection
- Price data (SOL/USD)
- Market cap

## 📄 File Descriptions

### **🔥 Core Scripts**

#### `pf_tf.py` - Moralis Token Fetcher

**Purpose:** Fetches fresh pump.fun tokens using Moralis API

```bash
# Usage
python pf_tf.py --minutes 30 --limit 50
```

- Fetches tokens launched in last X minutes
- Filters for trackable tokens (< 45min old)
- Saves to `pump_tokens.csv`
- Backs up previous tokens to `backlog_pump_tokens.csv`

#### `get_fresh_tokens.py` - RPC Token Fetcher

**Purpose:** Alternative token fetcher using Solana RPC (no API key needed)

```bash
# Usage
python get_fresh_tokens.py
```

- Extracts tokens from recent pump.fun transactions
- Works when Moralis API is down
- Same backup/archive system as `pf_tf.py`

#### `pf_tracker.py` - Main Tracking Engine

**Purpose:** Tracks tokens from CSV and collects real-time metrics

```bash
# Usage
python pf_tracker.py
```

- Reads tokens from `pump_tokens.csv`
- Tracks each token for 45 minutes max
- Saves individual tracking files to `tracks/`
- Uses only Solana RPC + CoinGecko (no Moralis dependency)

### **🔧 Utility Scripts**

#### `debug_transactions.py` - Transaction Debugger

**Purpose:** Debug why certain tokens aren't showing activity

```bash
# Usage
python debug_transactions.py
```

- Analyzes transaction history for test tokens
- Helps troubleshoot tracking issues
- Shows pump.fun program interactions

#### `test_sources.py` - API Testing Tool

**Purpose:** Test all data sources to verify connectivity

```bash
# Usage
python test_sources.py
```

- Tests Solana RPC connectivity
- Verifies CoinGecko price API
- Checks Jupiter/DexScreener price sources

### **📊 Data Files**

#### `pump_tokens.csv` - Active Token List

Contains tokens currently being tracked:

```csv
tokenAddress,name,symbol,createdAt,age_minutes,trackable,market_cap_usd
2DyxvdkE...,Token Name,SYM,2025-06-02T01:36:29+00:00,5.2,true,1000
```

#### `backlog_pump_tokens.csv` - Token Archive

Historical archive of all discovered tokens (grows over time)

#### `tracks/` Directory

Individual CSV files for each tracked token:

```csv
timestamp,mint,progress,R,A,unique_buyers,top3_pct,whale_flag,...
2025-06-02T00:32:05+00:00,Dqy5a8...,0,0.0,0.0,0,0.0,False,...
```

## 🚀 Quick Start Guide

### **1. Installation**

```bash
# Clone/download the project
cd pump_tracker
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### **2. Get Fresh Tokens (Choose One Method)**

**Method A: Using Moralis API (Recommended)**

```bash
# Set your Moralis API key in pf_tf.py (line 20)
# Then fetch tokens
python pf_tf.py --minutes 30 --limit 50
```

**Method B: Using RPC (No API key needed)**

```bash
# Works without any setup
python get_fresh_tokens.py
```

### **3. Start Tracking**

```bash
# Track all tokens from pump_tokens.csv
python pf_tracker.py
```

### **4. Monitor Results**

- **Live logs:** Watch console output
- **Individual tracks:** Check `tracks/` directory
- **System logs:** Check `logs/` directory

## 🔄 Typical Workflow

```bash
# 1. Get fresh tokens (every 30 minutes)
python pf_tf.py --minutes 30 --limit 50

# 2. Start tracking
python pf_tracker.py

# 3. Monitor output
tail -f logs/rpc_tracker_20250602.log

# 4. Check results
ls tracks/
cat tracks/TokenName_20250602_013430.csv
```

## 📊 Understanding the Output

### **Console Output:**

```
🚀 Starting tracking for 3 tokens...
📡 Started tracking Token ABC (2DyxvdkE) for 44.6min → tracks/2DyxvdkE_20250602_013430.csv
✅ Token ABC | Progress: 2.5% | R: 1.4312 | Buyers: 2 | Price: $0.00000443
🎯 Token ABC | Found 3 transactions!
```

### **CSV Data Format:**

- **timestamp:** When measurement was taken
- **progress:** Bonding curve progress (0-100%)
- **R:** Cumulative buy-sell difference (SOL)
- **A:** Change in R from previous measurement
- **unique_buyers:** Number of different buyers
- **whale_flag:** True if >1 SOL transaction detected

## ⚙️ Configuration

### **Key Settings (in pf_tracker.py):**

```python
WINDOW = 10        # Sample every 10 seconds
DURATION = 45 * 60 # Track for 45 minutes max
TARGET_SOL = 85    # Bonding curve target
```

### **Moralis API Setup:**

1. Get API key from [Moralis.io](https://moralis.io)
2. Replace key in `pf_tf.py` line 20
3. Choose Solana network in dashboard

## 🔍 Troubleshooting

### **No tokens found:**

```bash
# Test data sources
python test_sources.py

# Debug specific token
python debug_transactions.py
```

### **Rate limiting errors:**

- Switch to RPC fetcher: `python get_fresh_tokens.py`
- Reduce fetch frequency

### **No trading activity:**

- Tokens may be inactive (normal for older tokens)
- Use fresher tokens (< 10 minutes old)

## 📈 Data Analysis

### **Finding Successful Tokens:**

```bash
# Look for tokens with high progress
grep "progress.*[5-9][0-9]" tracks/*.csv

# Find whale activity
grep "whale_flag.*True" tracks/*.csv

# Check for bot activity
grep "bot_like.*True" tracks/*.csv
```

### **Performance Metrics:**

- **Progress > 10%:** Token gaining traction
- **Unique buyers > 5:** Good community interest
- **Top3_pct < 80%:** Not whale-dominated
- **Whale_flag = True:** Large investor interest

## 🤝 Contributing

1. Test your changes with `python test_sources.py`
2. Ensure logs are clear and informative
3. Maintain the backup/archive system
4. Update this README if adding new features

---

**🎯 Goal:** Track pump.fun tokens efficiently and identify trending/successful launches early through comprehensive metrics analysis.

```python
#!/usr/bin/env python3
#!/usr/bin/env python3
"""
📡 Pump.fun Tracker  -  Authentic PDA Edition (FULLY FIXED)
• Tracks any Pump.fun launch until 45 min or migration
• Metrics every 10 s, written one-row-per-token-per-interval
• Uses only Solana RPC + CoinGecko
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

# ────────────────── constants ──────────────────
HELIUS_KEY = os.getenv("HELIUS_KEY")
if not HELIUS_KEY:
    print("❌ ERROR: HELIUS_KEY environment variable not set!")
    sys.exit(1)

SOLANA_RPC_URL = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_KEY}"
HELIUS_WS_URL = f"wss://mainnet.helius-rpc.com/?api-key={HELIUS_KEY}"
COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd"
TOKEN_LIST_CSV = "pump_tokens.csv"
CONSOLIDATED_CSV = "all_token_tracks.csv"

PUMP_PROGRAM_ID = PublicKey("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")
BONDING_SEED = b"bonding-curve"
MIGRATION_HELPER = "39azUYFWPz3VHgKCf3VChUwbpURdCHRxjWVowf5jUJjg"

INITIAL_RTOK_RES = 793_100_000 * 1_000_000      # realTokenReserves at launch
TOTAL_SUPPLY = 1_000_000_000
WINDOW = 10                            # seconds
DURATION = 45 * 60                       # seconds
MIN_VOL = 0.00005                       # SOL

CSV_HEADER = [
    "timestamp", "mint", "sol_in_pool", "sol_raised_total", "progress",
    "sol_flow_10s", "sol_acceleration", "unique_buyers", "top3_pct",
    "lp_burn", "whale_flag", "buy_sell_delta", "buy_pressure", "sell_pressure", "bot_like"
]

# ────────────────── helpers ──────────────────


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def curve_pda(mint: str) -> str:
    # Convert the mint string to a PublicKey then to bytes via bytes()
    mint_bytes = bytes(PublicKey(mint))
    return str(PublicKey.find_program_address(
        [BONDING_SEED, mint_bytes],
        PUMP_PROGRAM_ID
    )[0])

# ────────────────── dataclasses ──────────────────


@dataclass
class IntervalState:
    start: datetime
    last_R: float = 0.0
    last_sol: float = 0.0
    last_buy_vol: float = 0.0
    last_sell_vol: float = 0.0


@dataclass
class SamplerTask:
    mint: str
    name: str
    launch_time: datetime
    curve: str
    state: IntervalState
    remaining_duration: float
    lp_mint: Optional[str] = None
    migration_event: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class Transaction:
    signature: str
    signer: str
    amount_sol: float
    timestamp: float
    is_buy: bool

# ────────────────── tracker class ──────────────────


class RPCTracker:
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.log = self._init_logger()
        self.active_tasks: Dict[str, SamplerTask] = {}
        self.sol_price_cache: Dict[int, float] = {}
        self.csv_lock = threading.Lock()  # Thread-safe CSV writing

        if not os.path.exists(CONSOLIDATED_CSV):
            with open(CONSOLIDATED_CSV, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=CSV_HEADER).writeheader()

    # -------------- bootstrap --------------
    def _init_logger(self):
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[logging.FileHandler(f"logs/tracker_{time.strftime('%Y%m%d')}.log"),
                      logging.StreamHandler(sys.stdout)]
        )
        return logging.getLogger("tracker")

    async def run(self):
        async with aiohttp.ClientSession() as sess:
            self.session = sess
            self.log.info("✅ RPC connection established")

            tokens = self._load_tokens()
            if not tokens:
                self.log.error("No tokens to track.")
                return

            # spawn sampler per token
            for t in tokens:
                await self._start_sampler(t)

            # housekeeping & heartbeat
            await asyncio.gather(
                self._monitor_tasks(),
                self._migration_listener(),
                self._heartbeat(),
                self._cache_cleanup(),
                return_exceptions=True
            )

    # -------------- CSV --------------------
    def _load_tokens(self):
        out = []
        if not os.path.exists(TOKEN_LIST_CSV):
            self.log.error("token CSV missing.")
            return out
        try:
            with open(TOKEN_LIST_CSV, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row.get("trackable", "").lower() == "true":
                        out.append(row)
        except Exception as e:
            self.log.error(f"Error reading token CSV: {e}")
            return []
        self.log.info(f"Loaded {len(out)} trackable tokens.")
        return out

    def _csv_append(self, row: Dict):
        """Thread-safe CSV writing"""
        try:
            with self.csv_lock:
                with open(CONSOLIDATED_CSV, "a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=CSV_HEADER).writerow(row)
        except Exception as e:
            logging.getLogger("tracker").error(f"Failed to write CSV: {e}")

    # -------------- sampler ----------------
    async def _start_sampler(self, token: Dict):
        mint = token["tokenAddress"]
        name = token.get("name", "Unknown")
        try:
            launch = datetime.fromisoformat(
                token["createdAt"].replace("Z", "+00:00"))
        except ValueError as e:
            self.log.error(f"Invalid date format for {name}: {e}")
            return

        age = (datetime.now(timezone.utc)-launch).total_seconds()
        if age > DURATION:
            self.log.info(f"⏩ {name} too old, skip.")
            return

        st = SamplerTask(
            mint=mint, name=name, launch_time=launch,
            curve=curve_pda(mint),
            state=IntervalState(start=datetime.now(timezone.utc)),
            remaining_duration=DURATION-age
        )
        self.active_tasks[mint] = st
        asyncio.create_task(self._sampler(st))
        self.log.info(
            f"📡 started {name} {mint[:8]} for {st.remaining_duration/60:.1f} min")

    def _default_row(self, mint: str) -> Dict:
        return {
            "timestamp": now_iso(),
            "mint": mint,
            "sol_in_pool": 0,
            "sol_raised_total": 0,
            "progress": 0,
            "sol_flow_10s": 0,
            "sol_acceleration": 0,
            "unique_buyers": 0,
            "top3_pct": 0,
            "lp_burn": False,
            "whale_flag": False,
            "buy_sell_delta": 0,
            "buy_pressure": 0,
            "sell_pressure": 0,
            "bot_like": False
        }

    async def _sampler(self, t: SamplerTask):
        end = datetime.now(timezone.utc) + \
            timedelta(seconds=t.remaining_duration)
        while datetime.now(timezone.utc) < end and not t.migration_event.is_set():
            try:
                row = await self._collect_metrics(t)
            except Exception as err:
                self.log.error(f"{t.name} window failed: {err}")
                row = self._default_row(t.mint)
            self._csv_append(row)
            # Update state properly
            t.state.last_R = row["sol_flow_10s"]
            t.state.last_buy_vol = row["buy_pressure"] * WINDOW
            t.state.last_sell_vol = row["sell_pressure"] * WINDOW
            await asyncio.sleep(WINDOW)
        self.active_tasks.pop(t.mint, None)
        self.log.info(f"✅ completed {t.name}")

    # -------------- collect ----------------
    async def _collect_metrics(self, t: SamplerTask) -> Dict:
        curve_state = await self._get_curve_state(t.curve)

        # Direct SOL pool monitoring - much more reliable!
        sol_in_pool = curve_state["v_sol"] / 1e9  # Convert lamports to SOL
        sol_raised_total = curve_state["r_sol"] / 1e9  # Total SOL raised

        # Progress based on token reserves (closer to migration)
        progress = max(
            0, min(100, 100 - (curve_state["r_tok"]/INITIAL_RTOK_RES)*100))

        # SOL flow metrics (much more meaningful than price-based)
        sol_flow_10s = sol_raised_total - t.state.last_sol
        sol_acceleration = sol_flow_10s - t.state.last_R
        t.state.last_sol = sol_raised_total
        t.state.last_R = sol_flow_10s

        # Transaction analysis for trading behavior
        txs = await self._recent_transactions(t.mint, t.curve)
        tx_metrics = self._process_transactions(txs)
        buy_sell_delta = tx_metrics["buy_volume"] - tx_metrics["sell_volume"]

        # LP burn check
        lp_burn = await self._check_lp_burn(t.lp_mint or t.mint)

        return {
            "timestamp": now_iso(),
            "mint": t.mint,
            # Current tradeable SOL
            "sol_in_pool": round(sol_in_pool, 6),
            "sol_raised_total": round(sol_raised_total, 6),  # Total SOL raised
            "progress": round(progress, 2),                  # % to migration
            "sol_flow_10s": round(sol_flow_10s, 6),         # SOL change in 10s
            # Change in flow rate
            "sol_acceleration": round(sol_acceleration, 6),
            "unique_buyers": tx_metrics["unique_buyers"],
            "top3_pct": round(tx_metrics["top3_pct"], 2),
            "lp_burn": lp_burn,
            "whale_flag": tx_metrics["whale_flag"],
            "buy_sell_delta": round(buy_sell_delta, 4),
            "buy_pressure": round(tx_metrics["buy_volume"]/WINDOW, 5),
            "sell_pressure": round(tx_metrics["sell_volume"]/WINDOW, 5),
            "bot_like": tx_metrics["bot_like"]
        }

    def _calculate_price(self, curve_state: Dict) -> float:
        """More robust price calculation"""
        v_tok = curve_state.get("v_tok", 0)
        v_sol = curve_state.get("v_sol", 0)

        if v_tok == 0 or v_sol == 0:
            return 0.0

        # Use constant product formula: price = SOL_reserves / TOKEN_reserves
        return v_sol / v_tok

    # -------------- curve state ------------
    async def _get_curve_state(self, curve: str) -> Dict:
        payload = {"jsonrpc": "2.0", "id": 1, "method": "getAccountInfo",
                   "params": [curve, {"encoding": "base64"}]}
        resp = await self._rpc_post(payload)
        value = resp.get("result", {}).get("value")
        if not value or not value.get("data"):
            # curve PDA not initialised yet – return zeros so metrics stay 0
            return dict(v_tok=1, v_sol=0, r_tok=INITIAL_RTOK_RES,
                        r_sol=0, complete=False)
        data64 = value["data"][0]
        buf = base64.b64decode(data64)

        try:
            return self._parse_curve_data(buf)
        except Exception as e:
            self.log.error(f"Failed to parse curve data: {e}")
            return dict(v_tok=1, v_sol=0, r_tok=INITIAL_RTOK_RES,
                        r_sol=0, complete=False)

    def _parse_curve_data(self, buf: bytes) -> Dict:
        """More robust curve data parsing"""
        if len(buf) < 16:  # Minimum size check
            return dict(v_tok=1, v_sol=0, r_tok=INITIAL_RTOK_RES,
                        r_sol=0, complete=False)

        # Try different parsing strategies
        try:
            # Strategy 1: Original format
            if len(buf) >= 81:
                data = buf[8:81]
                v_tok, v_sol, r_tok, r_sol, supply, complete = struct.unpack(
                    "<QQQQQ?32x", data)
                return dict(v_tok=v_tok, v_sol=v_sol, r_tok=r_tok,
                            r_sol=r_sol, complete=complete)
        except struct.error:
            pass

        try:
            # Strategy 2: Without padding
            if len(buf) >= 49:
                data = buf[8:49]
                v_tok, v_sol, r_tok, r_sol, supply, complete = struct.unpack(
                    "<QQQQQ?", data)
                return dict(v_tok=v_tok, v_sol=v_sol, r_tok=r_tok,
                            r_sol=r_sol, complete=complete)
        except struct.error:
            pass

        try:
            # Strategy 3: Just the essential fields
            if len(buf) >= 40:
                data = buf[8:40]
                v_tok, v_sol, r_tok, r_sol = struct.unpack("<QQQQ", data)
                return dict(v_tok=v_tok, v_sol=v_sol, r_tok=r_tok,
                            r_sol=r_sol, complete=False)
        except struct.error:
            pass

        # Fallback: return safe defaults
        return dict(v_tok=1, v_sol=0, r_tok=INITIAL_RTOK_RES,
                    r_sol=0, complete=False)

    # -------------- tx helpers -------------
    async def _recent_transactions(self, mint: str, curve: str) -> List[Transaction]:
        """light 20-sig scan on the curve PDA for bot / buyer metrics"""
        try:
            payload = {
                "jsonrpc": "2.0", "id": 1, "method": "getSignaturesForAddress",
                "params": [curve, {"limit": 20}]
            }
            async with self.session.post(SOLANA_RPC_URL, json=payload) as r:
                result = await r.json()
                if "error" in result:
                    self.log.error(
                        f"Error getting signatures: {result['error']}")
                    return []
                sigs = [s["signature"] for s in result.get("result", [])]
        except Exception as e:
            self.log.error(f"Failed to get signatures: {e}")
            return []

        txs = []
        for sig in sigs:
            tx = await self._get_tx_details(sig, mint)
            if tx:
                txs.append(tx)
        return txs

    async def _get_tx_details(self, sig: str, mint: str) -> Optional[Transaction]:
        try:
            payload = {
                "jsonrpc": "2.0", "id": 1, "method": "getTransaction",
                "params": [sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}]
            }
            async with self.session.post(SOLANA_RPC_URL, json=payload) as r:
                result = await r.json()
                data = result.get("result")

            if not data or data.get("meta", {}).get("err") is not None:
                return None

            return self._parse_transaction(data, mint)

        except Exception as e:
            self.log.error(f"Failed to get tx details for {sig}: {e}")
            return None

    def _parse_transaction(self, data: Dict, mint: str) -> Optional[Transaction]:
        """More robust transaction parsing"""
        try:
            msg = data["transaction"]["message"]
            pre = data["meta"]["preBalances"]
            post = data["meta"]["postBalances"]

            # Look for the signer (first account that signed)
            signer_idx = None
            signer_pubkey = None

            for i, acc in enumerate(msg["accountKeys"]):
                if acc.get("signer", False):
                    signer_idx = i
                    signer_pubkey = acc["pubkey"]
                    break

            if signer_idx is None or signer_idx >= len(pre) or signer_idx >= len(post):
                return None

            # Calculate SOL balance change for the signer
            sol_diff = (post[signer_idx] - pre[signer_idx]) / 1e9

            if abs(sol_diff) < MIN_VOL:
                return None

            # Correct buy/sell logic
            # If signer's SOL balance decreased, they bought tokens (spent SOL)
            # If signer's SOL balance increased, they sold tokens (received SOL)
            is_buy = sol_diff < 0

            return Transaction(
                signature=data.get("transaction", {}).get(
                    "signatures", [""])[0],
                signer=signer_pubkey,
                amount_sol=abs(sol_diff),
                timestamp=data.get("blockTime", 0),
                is_buy=is_buy
            )

        except Exception as e:
            self.log.error(f"Error parsing transaction: {e}")
            return None

    def _process_transactions(self, txs: List[Transaction]) -> Dict:
        """Improved transaction processing with better calculations"""
        buy_vol = sell_vol = 0.0
        buyers = {}
        sellers = {}

        for tx in txs:
            if tx.is_buy:
                buy_vol += tx.amount_sol
                buyers[tx.signer] = buyers.get(tx.signer, 0) + tx.amount_sol
            else:
                sell_vol += tx.amount_sol
                sellers[tx.signer] = sellers.get(tx.signer, 0) + tx.amount_sol

        unique = len(set(buyers.keys()) | set(sellers.keys()))

        # FIXED: Better top3 calculation with proper bounds checking
        if buy_vol > 0 and buyers:
            top3_amounts = sorted(buyers.values(), reverse=True)[:3]
            top3_pct = min(100.0, sum(top3_amounts) / buy_vol * 100)
        else:
            top3_pct = 0.0

        whale = any(tx.amount_sol >= 1.0 for tx in txs)
        bot = self._detect_bot(txs)

        return dict(
            buy_volume=buy_vol,
            sell_volume=sell_vol,
            unique_buyers=len(buyers),
            top3_pct=top3_pct,
            whale_flag=whale,
            bot_like=bot
        )

    def _detect_bot(self, txs: List[Transaction]) -> bool:
        """FIXED: Improved bot detection with better thresholds"""
        if len(txs) < 3:  # Lowered threshold from 4 to 3
            return False

        # Look for rapid-fire small transactions
        buys = [tx for tx in txs if tx.is_buy and tx.amount_sol < 0.1]
        if len(buys) < 3:  # Lowered threshold
            return False

        buys.sort(key=lambda x: x.timestamp)

        # FIXED: More reasonable timing - 3+ transactions within 2 seconds
        for i in range(len(buys) - 2):  # Check for 3 consecutive txs
            if buys[i + 2].timestamp - buys[i].timestamp <= 2.0:  # 2 second window
                return True

        # Also check for suspicious patterns (same amounts, regular intervals)
        # Round to avoid float precision issues
        amounts = [round(tx.amount_sol, 6) for tx in buys]
        if len(set(amounts)) == 1 and len(amounts) >= 3:  # All same amounts
            return True

        # FIXED: Check for regular intervals (potential bot pattern)
        if len(buys) >= 4:
            intervals = []
            for i in range(1, len(buys)):
                intervals.append(buys[i].timestamp - buys[i-1].timestamp)

            # If most intervals are very similar (within 0.5s), likely a bot
            if len(intervals) >= 3:
                avg_interval = sum(intervals) / len(intervals)
                similar_intervals = sum(
                    1 for i in intervals if abs(i - avg_interval) < 0.5)
                # 75% of intervals are similar
                if similar_intervals >= len(intervals) * 0.75:
                    return True

        return False

    # -------------- LP burn -----------------
    async def _check_lp_burn(self, lp_mint: str) -> bool:
        if not lp_mint:
            return False
        try:
            sigs_payload = {"jsonrpc": "2.0", "id": 1, "method": "getSignaturesForAddress",
                            "params": [lp_mint, {"limit": 5}]}
            async with self.session.post(SOLANA_RPC_URL, json=sigs_payload) as r:
                result = await r.json()
                if "error" in result:
                    return False
                sigs = [s["signature"] for s in result.get("result", [])]

            for sig in sigs:
                tx_payload = {"jsonrpc": "2.0", "id": 1, "method": "getTransaction",
                              "params": [sig, {"encoding": "jsonParsed"}]}
                async with self.session.post(SOLANA_RPC_URL, json=tx_payload) as r:
                    result = await r.json()
                    if "error" in result or not result.get("result"):
                        continue
                    ins = result["result"]["transaction"]["message"]["instructions"]
                if any(insn.get("parsed", {}).get("type") == "burn" for insn in ins):
                    return True
        except Exception as e:
            self.log.error(f"Error checking LP burn: {e}")
        return False

    # -------------- price -------------------
    async def _get_sol_price(self) -> float:
        minute = int(time.time()/60)
        if minute in self.sol_price_cache:
            return self.sol_price_cache[minute]
        try:
            async with self.session.get(COINGECKO_URL) as r:
                data = await r.json()
                price = data["solana"]["usd"]
            self.sol_price_cache[minute] = price
            return price
        except Exception as e:
            self.log.error(f"Failed to get SOL price: {e}")
            # Return cached price if available, otherwise default
            if self.sol_price_cache:
                return list(self.sol_price_cache.values())[-1]
            return 100.0  # fallback price

    async def _cache_cleanup(self):
        """Clean up old cache entries every hour"""
        while True:
            await asyncio.sleep(3600)  # 1 hour
            current_minute = int(time.time() / 60)
            # Keep only last 24 hours of cache
            cutoff = current_minute - (24 * 60)
            self.sol_price_cache = {
                k: v for k, v in self.sol_price_cache.items() if k > cutoff
            }

    # -------------- migration listener ------
    async def _migration_listener(self):
        max_retries = 5
        retry_count = 0

        while retry_count < max_retries:
            try:
                async with self.session.ws_connect(HELIUS_WS_URL) as ws:
                    await ws.send_json({
                        "jsonrpc": "2.0", "id": 1, "method": "logsSubscribe",
                        "params": [{"mentions": [MIGRATION_HELPER]},
                                   {"commitment": "confirmed", "encoding": "jsonParsed"}]
                    })

                    retry_count = 0  # Reset on successful connection

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = msg.json()
                            # just the {"result":...} handshake
                            if "params" not in data:
                                continue

                            # FIXED: More robust migration detection
                            try:
                                result = data["params"]["result"]
                                accs = result["value"]["transaction"]["message"]["accountKeys"]

                                # FIXED: Better account parsing - search all accounts for our tracked mints
                                found_mint = None
                                potential_lp_mint = None

                                for i, acc in enumerate(accs):
                                    acc_key = acc if isinstance(
                                        acc, str) else acc.get("pubkey", "")

                                    # Check if this account is one of our tracked tokens
                                    if acc_key in self.active_tasks:
                                        found_mint = acc_key
                                        # Try to find LP mint in nearby accounts (flexible positioning)
                                        # Check ±2 positions
                                        for j in range(max(0, i-2), min(len(accs), i+3)):
                                            if j != i:
                                                potential_acc = accs[j] if isinstance(
                                                    accs[j], str) else accs[j].get("pubkey", "")
                                                # Basic heuristic: LP mint addresses often start with different patterns
                                                if potential_acc and potential_acc != acc_key:
                                                    potential_lp_mint = potential_acc
                                        break

                                if found_mint:
                                    st = self.active_tasks[found_mint]
                                    if potential_lp_mint:
                                        st.lp_mint = potential_lp_mint
                                    st.migration_event.set()
                                    self.log.info(
                                        f"🎉 {st.name} migrated - stopping sampler.")

                            except Exception as e:
                                self.log.error(
                                    f"Error processing migration event: {e}")

                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            self.log.error(f"WebSocket error: {msg.data}")
                            break
                        elif msg.type == aiohttp.WSMsgType.CLOSE:
                            self.log.warning("WebSocket connection closed")
                            break

            except Exception as e:
                retry_count += 1
                self.log.error(
                    f"Migration listener error (attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    # Exponential backoff
                    await asyncio.sleep(min(2 ** retry_count, 30))
                else:
                    self.log.error("Migration listener failed permanently")
                    break

    # -------------- misc --------------------
    async def _monitor_tasks(self):
        while self.active_tasks:
            await asyncio.sleep(30)
        self.log.info("All samplers done.")

    async def _heartbeat(self):
        while True:
            self.log.info(f"⏰ heartbeat | active = {len(self.active_tasks)}")
            await asyncio.sleep(300)

    async def _rpc_post(self, payload, *, retries=3):
        for attempt in range(retries):
            try:
                async with self.session.post(SOLANA_RPC_URL, json=payload, timeout=10) as r:
                    data = await r.json()
                    if "error" in data:
                        raise RuntimeError(
                            f"RPC Error: {data['error'].get('message', 'Unknown error')}")
                    return data
            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as e:
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(0.5 * 2 ** attempt)


# ────────────────── main ──────────────────
if __name__ == "__main__":
    try:
        asyncio.run(RPCTracker().run())
    except KeyboardInterrupt:
        print("\n⏹ stopped")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


```
