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

```

#!/usr/bin/env python3
"""
📡 Pump.fun Tracker – dual-CSV, single-mint edition (IMPROVED)
• Candle rows every 1/5/10 s  ➜ all_token_tracks.csv
• One row per on-chain event ➜ all_token_events.csv
• Enhanced data accuracy and error handling
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
import json
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from solana.publickey import PublicKey   # pip install solana

# ───────── CLI argument (one mint per process) ─────────
import argparse
ap = argparse.ArgumentParser(description="Track a single Pump.fun mint")
ap.add_argument("--mint", required=True, help="Token mint address")
ap.add_argument("--config", default="config.json",
                help="Configuration file path")
ARGS = ap.parse_args()

# ───────── Load configuration ─────────


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file with corrected Pump.fun defaults"""
    defaults = {
        # FIXED: Correct Pump.fun bonding curve parameters
        # ~1.073B tokens (correct)
        "initial_token_reserves": 1073000000000000,
        # 30 SOL in lamports (correct)
        "initial_sol_reserves": 30000000000,
        # Virtual SOL reserves (30 SOL)
        "virtual_sol_reserves": 30000000000,
        "virtual_token_reserves": 1073000000000000,     # Virtual token reserves

        "min_volume_sol": 0.001,                        # Minimum volume threshold
        "tx_lookback_minutes": 5,                       # Transaction lookback window
        "max_tx_fetch": 50,                             # Max transactions to fetch
        # INCREASED: More time for RPC calls
        "rpc_timeout": 15,
        "websocket_reconnect_delay": 5,
        "data_validation_enabled": True,
        "progress_bounds_check": True,

        # NEW: Additional validation parameters
        # Flag flows > 50 SOL as suspicious
        "max_reasonable_sol_flow": 50,
        "curve_state_cache_seconds": 2,                 # Cache curve state briefly
        "enable_debug_logging": False                   # Enable detailed debug logs
    }

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                defaults.update(user_config)
        except Exception as e:
            print(f"Warning: Could not load config {config_path}: {e}")
    else:
        # Create default config file
        with open(config_path, 'w') as f:
            json.dump(defaults, f, indent=2)
        print(f"Created default config at {config_path}")

    return defaults


CONFIG = load_config(ARGS.config)

# ───────── constants ─────────
HELIUS_KEY = os.getenv("HELIUS_KEY")
if not HELIUS_KEY:
    print("❌ HELIUS_KEY env var not set")
    sys.exit(1)

RPC = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_KEY}"
WS = f"wss://mainnet.helius-rpc.com/?api-key={HELIUS_KEY}"

TRACKS_CSV = "all_token_tracks.csv"
EVENTS_CSV = "all_token_events.csv"
ERRORS_CSV = "data_errors.csv"

PUMP_PROGRAM_ID = PublicKey("6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P")
BONDING_SEED = b"bonding-curve"
MIGRATION_HELPER = "39azUYFWPz3VHgKCf3VChUwbpURdCHRxjWVowf5jUJjg"

DURATION_SECS = 45 * 60

TRACK_HEADER = [
    "timestamp", "mint", "sol_in_pool", "sol_raised_total", "progress",
    "sol_flow", "sol_accel", "unique_buyers", "top3_pct", "lp_burn",
    "whale_flag", "buy_sell_delta", "buy_pressure", "sell_pressure", "bot_like",
    "data_quality", "validation_errors"
]

EVENT_HEADER = [
    "slot", "block_time", "recv_time", "mint", "signature",
    "is_buy", "amount_sol", "v_sol", "v_tok", "buyer_pubkey", "data_lag_ms"
]

ERROR_HEADER = [
    "timestamp", "mint", "error_type", "error_message", "context"
]

# ───────── helpers ─────────


def now_iso(): return datetime.now(timezone.utc).isoformat()
def now_ts(): return datetime.now(timezone.utc).timestamp()


def curve_pda(mint: str) -> str:
    return str(PublicKey.find_program_address(
        [BONDING_SEED, bytes(PublicKey(mint))], PUMP_PROGRAM_ID)[0])

# ───────── dataclasses ─────────


@dataclass
class CurveState:
    """Represents bonding curve state with validation and caching"""
    v_tok: int
    v_sol: int
    r_tok: int
    r_sol: int
    timestamp: float = field(default_factory=time.time)
    is_valid: bool = True
    validation_issues: List[str] = field(default_factory=list)

    def validate(self, mint: str, logger) -> List[str]:
        """Enhanced validation with more comprehensive checks"""
        issues = []

        # Basic value checks
        if self.v_tok <= 0:
            issues.append("virtual_token_reserves_zero_or_negative")
        if self.v_sol < 0:
            issues.append("virtual_sol_reserves_negative")
        if self.r_tok < 0:
            issues.append("real_token_reserves_negative")
        if self.r_sol < 0:
            issues.append("real_sol_reserves_negative")

        # Reasonable bounds checks with correct values
        if self.r_tok > CONFIG["initial_token_reserves"]:
            issues.append("token_reserves_exceed_initial")

        # Check for total token consistency
        total_tokens = self.v_tok + self.r_tok
        if total_tokens > CONFIG["initial_token_reserves"] * 1.1:  # Allow 10% variance
            issues.append("total_tokens_inconsistent")

        # Reasonable SOL bounds - virtual + real shouldn't exceed reasonable limits
        total_sol = self.v_sol + self.r_sol
        if total_sol > CONFIG["virtual_sol_reserves"] * 20:  # Allow up to 600 SOL total
            issues.append("total_sol_reserves_unreasonably_high")

        # Check for impossible states
        # Virtual SOL shouldn't exceed initial + 10%
        if self.v_sol > CONFIG["virtual_sol_reserves"] * 1.1:
            issues.append("virtual_sol_exceeds_initial")

        if issues:
            logger.warning(f"Curve validation issues for {mint}: {issues}")
            self.is_valid = False
            self.validation_issues = issues

        return issues


@dataclass
class IntervalState:
    start: datetime
    last_R: float = 0.
    last_sol: float = 0.
    last_curve_state: Optional[CurveState] = None


@dataclass
class SamplerTask:
    mint: str
    name: str
    launch: datetime
    curve: str
    state: IntervalState
    initial_curve_state: Optional[CurveState] = None
    lp_mint: Optional[str] = None
    migration_event: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class Tx:
    sig: str
    signer: str
    sol: float
    ts: int
    is_buy: bool
    block_time: Optional[int] = None

# ───────── tracker ─────────


class Tracker:
    def __init__(self):
        self.log = self._init_logger()
        self.session: Optional[aiohttp.ClientSession] = None
        self.csv_lock = threading.Lock()
        self.active: Dict[str, SamplerTask] = {}
        self.websocket_task: Optional[asyncio.Task] = None
        self.reconnect_count = 0

        # Initialize CSV files
        for fn, h in [(TRACKS_CSV, TRACK_HEADER), (EVENTS_CSV, EVENT_HEADER), (ERRORS_CSV, ERROR_HEADER)]:
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

    def _log_error(self, mint: str, error_type: str, message: str, context: dict = None):
        """Log error to both logger and CSV"""
        self.log.error(f"{mint} | {error_type}: {message}")
        error_row = {
            "timestamp": now_iso(),
            "mint": mint,
            "error_type": error_type,
            "error_message": message,
            "context": json.dumps(context or {})
        }
        self._csv_write(ERRORS_CSV, error_row, ERROR_HEADER)

    # ─────── main entry ───────
    async def run(self):
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=CONFIG["rpc_timeout"])) as sess:
            self.session = sess
            self.log.info(f"RPC ready with config: {CONFIG}")

            # build metadata for the single mint supplied on CLI
            mint_meta = {
                "tokenAddress": ARGS.mint,
                "name": ARGS.mint[:8],
                "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "trackable": True,
            }
            await self._start(mint_meta)

            await asyncio.gather(
                self._monitor(), self._heartbeat(), self._event_hub_with_reconnect(),
                return_exceptions=True)

    # ─────── bootstrap one sampler ───────
    async def _start(self, row):
        mint, rowname = row["tokenAddress"], row.get("name", "?")
        launch = datetime.fromisoformat(
            row["createdAt"].replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc)-launch).total_seconds()
        if age > DURATION_SECS:
            return self.log.info(f"{rowname} too old, skip")

        # Get initial curve state with validation
        curve_addr = curve_pda(mint)
        try:
            initial_curve = await self._get_curve_validated(curve_addr, mint)
            if not initial_curve.is_valid:
                self._log_error(mint, "INVALID_INITIAL_STATE",
                                "Initial curve state failed validation",
                                {"curve_state": vars(initial_curve)})

            # Calculate initial SOL raised using validated state
            initial_raised = self._calculate_sol_raised(initial_curve)

            st = SamplerTask(
                mint, rowname, launch, curve_addr,
                IntervalState(datetime.now(timezone.utc),
                              last_sol=initial_raised),
                initial_curve_state=initial_curve
            )
            self.active[mint] = st
            asyncio.create_task(self._sampler(st))
            self.log.info(
                f"📡 {rowname[:12]}… sampler started with initial_raised={initial_raised:.6f}")

        except Exception as e:
            self._log_error(mint, "BOOTSTRAP_FAILED", str(e), {"mint": mint})
            raise

    def _calculate_sol_raised(self, curve_state: CurveState) -> float:
        """FIXED: Calculate total SOL raised from curve state"""
        if not curve_state.is_valid:
            return 0.0

        # CORRECTED LOGIC: For Pump.fun bonding curves:
        # - v_sol starts at ~30 SOL and decreases as people buy tokens
        # - SOL raised = initial virtual SOL - current virtual SOL
        initial_virtual_sol = CONFIG["virtual_sol_reserves"]
        current_virtual_sol = curve_state.v_sol

        sol_raised_lamports = initial_virtual_sol - current_virtual_sol
        sol_raised = sol_raised_lamports / 1e9

        # FIXED: Ensure non-negative and add bounds check
        sol_raised = max(0, sol_raised)

        # Sanity check: if SOL raised exceeds reasonable amount, flag it
        if sol_raised > 100:  # More than 100 SOL raised seems high for most tokens
            self.log.warning(f"Unusually high SOL raised: {sol_raised:.4f}")

        return sol_raised

    # ─────── sampler (candle writer) ───────
    async def _sampler(self, st: SamplerTask):
        consecutive_errors = 0
        max_consecutive_errors = 5

        while not st.migration_event.is_set():
            try:
                age = (datetime.now(timezone.utc)-st.launch).total_seconds()
                prog = await self._progress_pct_validated(st)
                window = 1 if (age < 120 or prog <
                               90) else 5 if age < 600 else 10

                row = await self._collect(st)
                self._csv_write(TRACKS_CSV, row, TRACK_HEADER)
                consecutive_errors = 0  # Reset error counter on success

            except Exception as e:
                consecutive_errors += 1
                self._log_error(st.mint, "SAMPLER_ERROR", str(e),
                                {"consecutive_errors": consecutive_errors})

                if consecutive_errors >= max_consecutive_errors:
                    self.log.error(
                        f"Too many consecutive errors for {st.name}, stopping sampler")
                    break

                # Exponential backoff
                await asyncio.sleep(min(window * (2 ** consecutive_errors), 60))
                continue

            await asyncio.sleep(window)

        self.active.pop(st.mint, None)
        self.log.info(f"✅ sampler done {st.name}")

    # ─────── enhanced curve state fetching ───────
    async def _get_curve_validated(self, addr: str, mint: str) -> CurveState:
        """Get curve state with validation"""
        try:
            raw_state = await self._get_curve_raw(addr)
            curve_state = CurveState(**raw_state)
            curve_state.validate(mint, self.log)
            return curve_state
        except Exception as e:
            self._log_error(mint, "CURVE_FETCH_ERROR",
                            str(e), {"curve_addr": addr})
            # Return safe default state
            return CurveState(
                v_tok=1, v_sol=0,
                r_tok=CONFIG["initial_token_reserves"],
                r_sol=0, is_valid=False
            )

    async def _get_curve_raw(self, addr: str) -> dict:
        """FIXED: Get raw curve state with better parsing and error handling"""
        payload = {"jsonrpc": "2.0", "id": 1, "method": "getAccountInfo",
                   "params": [addr, {"encoding": "base64"}]}

        try:
            response = await self._rpc(payload)
            val = response.get("result", {}).get("value")

            if not val or not val.get("data"):
                self.log.warning(f"No curve data found for {addr}")
                return self._get_default_curve_state()

            buf = base64.b64decode(val["data"][0])

            if CONFIG.get("enable_debug_logging", False):
                self.log.debug(f"Curve buffer length: {len(buf)}")
                if len(buf) >= 16:
                    self.log.debug(f"First 16 bytes: {buf[:16].hex()}")

            if len(buf) >= 81:
                try:
                    # Strategy 1: Original approach (skip 8-byte discriminator)
                    v_tok, v_sol, r_tok, r_sol = struct.unpack(
                        "<QQQQ", buf[8:40])
                    if self._validate_raw_curve_values(v_tok, v_sol, r_tok, r_sol):
                        return dict(v_tok=v_tok, v_sol=v_sol, r_tok=r_tok, r_sol=r_sol)
                except struct.error as e:
                    self.log.error(f"Struct unpacking error (strategy 1): {e}")

            if len(buf) >= 73:
                try:
                    # Strategy 2: Try different offset
                    v_tok, v_sol, r_tok, r_sol = struct.unpack(
                        "<QQQQ", buf[0:32])
                    if self._validate_raw_curve_values(v_tok, v_sol, r_tok, r_sol):
                        return dict(v_tok=v_tok, v_sol=v_sol, r_tok=r_tok, r_sol=r_sol)
                except struct.error:
                    pass

        except Exception as e:
            self.log.error(f"Error fetching curve state for {addr}: {e}")

        return self._get_default_curve_state()

    def _validate_raw_curve_values(self, v_tok: int, v_sol: int, r_tok: int, r_sol: int) -> bool:
        """FIXED: Validate raw curve values make sense"""
        if v_tok <= 0 or v_sol < 0 or r_tok < 0 or r_sol < 0:
            return False

        if v_tok > CONFIG["virtual_token_reserves"] * 2:
            return False

        if v_sol > CONFIG["virtual_sol_reserves"] * 2:
            return False

        if r_tok > CONFIG["initial_token_reserves"]:
            return False

        return True

    def _get_default_curve_state(self) -> dict:
        """FIXED: Return safe default curve state"""
        return dict(
            v_tok=CONFIG["virtual_token_reserves"],
            v_sol=CONFIG["virtual_sol_reserves"],
            r_tok=CONFIG["initial_token_reserves"],
            r_sol=0
        )

    # ─────── metric collection with validation ───────
    async def _collect(self, st: SamplerTask):
        """FIXED: Enhanced data collection with better validation"""
        curve_state = await self._get_curve_validated(st.curve, st.mint)
        validation_errors = []
        data_quality = "good"

        if not curve_state.is_valid:
            validation_errors.extend(curve_state.validation_issues)
            data_quality = "poor"

        # FIXED: Correct calculations
        sol_pool = curve_state.v_sol / 1e9
        total_sol_raised = self._calculate_sol_raised(curve_state)

        # FIXED: Progress calculation
        initial_tokens = CONFIG["initial_token_reserves"]
        remaining_tokens = curve_state.r_tok
        tokens_sold = initial_tokens - remaining_tokens
        prog = (tokens_sold / initial_tokens) * 100
        prog = max(0, min(100, prog))

        # FIXED: Flow calculation with better validation
        flow = total_sol_raised - st.state.last_sol

        # ENHANCED: More sophisticated flow validation
        max_reasonable_flow = CONFIG.get("max_reasonable_sol_flow", 50)
        if abs(flow) > max_reasonable_flow:
            validation_errors.append(
                f"excessive_flow_detected_{abs(flow):.2f}")
            data_quality = "questionable"
            self.log.warning(
                f"Large flow detected for {st.mint}: {flow:.4f} SOL")

        # FIXED: Acceleration calculation
        accel = flow - st.state.last_R

        # Update state
        st.state.last_sol = total_sol_raised
        st.state.last_R = flow
        st.state.last_curve_state = curve_state

        # Get transaction metrics with error handling
        try:
            txs = await self._recent_txs_bounded(st.curve, st.mint)
            tm = self._tx_metrics(txs)
        except Exception as e:
            self._log_error(st.mint, "TX_FETCH_ERROR", str(e))
            tm = {"buy_vol": 0, "sell_vol": 0, "unique": 0,
                  "top3_pct": 0, "whale": False, "bot": False}
            validation_errors.append("tx_fetch_failed")
            data_quality = "poor"

        delta = tm["buy_vol"] - tm["sell_vol"]

        # FIXED: Enhanced return with better rounding and validation
        return dict(
            timestamp=now_iso(),
            mint=st.mint,
            sol_in_pool=round(sol_pool, 6),
            sol_raised_total=round(total_sol_raised, 6),
            progress=round(prog, 2),
            sol_flow=round(flow, 6),
            sol_accel=round(accel, 6),
            unique_buyers=tm["unique"],
            top3_pct=round(tm["top3_pct"], 2),
            lp_burn=False,  # TODO: Implement LP burn detection
            whale_flag=tm["whale"],
            buy_sell_delta=round(delta, 4),
            buy_pressure=round(tm["buy_vol"], 5),
            sell_pressure=round(tm["sell_vol"], 5),
            bot_like=tm["bot"],
            data_quality=data_quality,
            validation_errors=";".join(
                validation_errors) if validation_errors else ""
        )

    async def _progress_pct_validated(self, st):
        """FIXED: Get progress percentage with correct calculation"""
        curve_state = await self._get_curve_validated(st.curve, st.mint)
        if not curve_state.is_valid:
            return 0.0

        # FIXED: Progress = (initial_tokens - remaining_tokens) / initial_tokens * 100
        initial_tokens = CONFIG["initial_token_reserves"]
        remaining_tokens = curve_state.r_tok
        tokens_sold = initial_tokens - remaining_tokens

        progress = (tokens_sold / initial_tokens) * 100

        # FIXED: Ensure bounds and add validation
        progress = max(0, min(100, progress))

        if CONFIG.get("enable_debug_logging", False):
            self.log.debug(
                f"Progress calc: initial={initial_tokens}, remaining={remaining_tokens}, progress={progress:.2f}%")

        return progress

    # ─────── enhanced transaction fetching ───────
    async def _recent_txs_bounded(self, curve: str, mint: str) -> List[Tx]:
        """Fetch recent transactions with time bounds and better error handling"""
        try:
            payload = {"jsonrpc": "2.0", "id": 1, "method": "getSignaturesForAddress",
                       "params": [curve, {"limit": CONFIG["max_tx_fetch"]}]}
            sigs_data = (await self._rpc(payload)).get("result", [])

            # Filter by time
            cutoff_time = time.time() - (CONFIG["tx_lookback_minutes"] * 60)
            recent_sigs = [s["signature"] for s in sigs_data
                           if s.get("blockTime", 0) > cutoff_time]

            out = []
            failed_fetches = 0
            for sig in recent_sigs:
                try:
                    tx = await self._get_tx_enhanced(sig)
                    if tx:
                        out.append(tx)
                except Exception as e:
                    failed_fetches += 1
                    if failed_fetches > 5:  # Stop if too many failures
                        self._log_error(mint, "TX_FETCH_LIMIT",
                                        f"Too many failed tx fetches: {failed_fetches}")
                        break

            return out
        except Exception as e:
            self._log_error(mint, "SIGNATURE_FETCH_ERROR", str(e))
            return []

    async def _get_tx_enhanced(self, sig: str) -> Optional[Tx]:
        """FIXED: Enhanced transaction fetching with better error handling"""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTransaction",
            "params": [
                sig,
                {
                    "encoding": "jsonParsed",
                    "maxSupportedTransactionVersion": 0,
                    "commitment": "confirmed"  # FIXED: Use confirmed for consistency
                }
            ]
        }

        try:
            data = (await self._rpc(payload)).get("result")
            if not data or data.get("meta", {}).get("err"):
                return None

            msg = data["transaction"]["message"]
            pre = data["meta"]["preBalances"]
            post = data["meta"]["postBalances"]

            # FIXED: Better signer detection
            signer_idx = None
            for i, acc in enumerate(msg["accountKeys"]):
                if isinstance(acc, dict) and acc.get("signer"):
                    signer_idx = i
                    break
                elif i == 0:  # First account is usually the signer
                    signer_idx = i
                    break

            if signer_idx is None or signer_idx >= len(pre):
                return None

            sol_diff = (post[signer_idx] - pre[signer_idx]) / 1e9

            # FIXED: Apply minimum volume filter
            if abs(sol_diff) < CONFIG["min_volume_sol"]:
                return None

            # FIXED: Better public key extraction
            if isinstance(msg["accountKeys"][signer_idx], dict):
                signer_pubkey = msg["accountKeys"][signer_idx]["pubkey"]
            else:
                signer_pubkey = msg["accountKeys"][signer_idx]

            return Tx(
                sig=sig,
                signer=signer_pubkey,
                sol=abs(sol_diff),
                ts=data.get("blockTime", 0),
                is_buy=sol_diff < 0,  # Buying costs SOL
                block_time=data.get("blockTime")
            )

        except Exception as e:
            self.log.warning(f"Failed to parse transaction {sig}: {e}")
            return None

    def _tx_metrics(self, txs: List[Tx]):
        """Calculate transaction metrics with enhanced bot detection"""
        buy = sell = 0
        buyers = {}
        rapid_txs = 0

        # Sort by timestamp for sequence analysis
        sorted_txs = sorted(txs, key=lambda x: x.ts or 0)

        for i, t in enumerate(sorted_txs):
            if t.is_buy:
                buy += t.sol
                buyers[t.signer] = buyers.get(t.signer, 0) + t.sol
            else:
                sell += t.sol

            # Check for rapid transactions (potential bot behavior)
            if i > 0 and t.ts and sorted_txs[i-1].ts:
                # Less than 2 seconds apart
                if abs(t.ts - sorted_txs[i-1].ts) < 2:
                    rapid_txs += 1

        top3 = sum(sorted(buyers.values(), reverse=True)
                   [:3]) / buy * 100 if buy else 0
        # More than 30% rapid transactions
        bot_like = rapid_txs > len(txs) * 0.3

        return dict(
            buy_vol=buy, sell_vol=sell, unique=len(buyers),
            top3_pct=top3, whale=any(t.sol >= 1 for t in txs),
            bot=bot_like
        )

    # ─────── WebSocket with reconnection ───────
    async def _event_hub_with_reconnect(self):
        """WebSocket event hub with automatic reconnection"""
        while self.active:
            try:
                await self._event_hub()
            except Exception as e:
                self.reconnect_count += 1
                self.log.error(
                    f"WebSocket error (reconnect #{self.reconnect_count}): {e}")

                if self.reconnect_count > 10:
                    self.log.error(
                        "Too many WebSocket reconnection attempts, giving up")
                    break

                delay = min(CONFIG["websocket_reconnect_delay"]
                            * (2 ** min(self.reconnect_count, 5)), 300)
                self.log.info(f"Reconnecting WebSocket in {delay} seconds...")
                await asyncio.sleep(delay)

    async def _event_hub(self):
        """FIXED: WebSocket event streaming with race condition mitigation"""
        if not self.active:
            return

        st = next(iter(self.active.values()))
        pda = st.curve

        async with self.session.ws_connect(WS) as ws:
            await ws.send_json({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "transactionSubscribe",
                "params": [
                    {"mentions": [pda]},
                    {
                        "commitment": "confirmed",  # FIXED: Use confirmed for faster notifications
                        "encoding": "jsonParsed",
                        "transactionDetails": "full"
                    }
                ]
            })

            self.log.info(f"WebSocket subscribed to {pda}")
            self.reconnect_count = 0

            async for msg in ws:
                if msg.type is not aiohttp.WSMsgType.TEXT:
                    break

                try:
                    j = msg.json()
                    if j.get("method") != "transactionNotification":
                        continue

                    recv_time = now_ts() * 1000
                    tx_obj = j["params"]["result"]["transaction"]
                    slot = j["params"]["result"]["context"]["slot"]
                    block_time = j["params"]["result"]["blockTime"]

                    # FIXED: Get curve state BEFORE processing to reduce race conditions
                    # This is still subject to race conditions but minimizes the window
                    pre_curve_state = await self._get_curve_validated(pda, st.mint)

                    # Validate transaction and extract details
                    acct_keys = tx_obj["message"]["accountKeys"]
                    if pda not in (a if isinstance(a, str) else a.get("pubkey") for a in acct_keys):
                        continue

                    signer_idx = next((i for i, acc in enumerate(acct_keys)
                                       if (isinstance(acc, dict) and acc.get("signer"))
                                       or (isinstance(acc, str) and i == 0)), None)

                    if signer_idx is None:
                        continue

                    pre = tx_obj["meta"]["preBalances"][signer_idx]
                    post = tx_obj["meta"]["postBalances"][signer_idx]
                    sol_diff = (post - pre) / 1e9

                    # FIXED: Better signer public key extraction
                    if isinstance(acct_keys[signer_idx], dict):
                        signer_pub = acct_keys[signer_idx]["pubkey"]
                    else:
                        signer_pub = acct_keys[signer_idx]

                    # FIXED: Skip dust transactions
                    if abs(sol_diff) < CONFIG["min_volume_sol"]:
                        continue

                    # Calculate data lag
                    data_lag = recv_time - \
                        (block_time * 1000) if block_time else 0

                    # FIXED: Use pre-transaction curve state for consistency
                    v_sol = pre_curve_state.v_sol / 1e9 if pre_curve_state.is_valid else 0
                    v_tok = pre_curve_state.v_tok / 1e9 if pre_curve_state.is_valid else 0

                    row = dict(
                        slot=slot,
                        block_time=block_time,
                        recv_time=now_iso(),
                        mint=st.mint,
                        signature=tx_obj["signatures"][0],
                        # Buying costs SOL (negative balance change)
                        is_buy=sol_diff < 0,
                        amount_sol=round(abs(sol_diff), 6),
                        v_sol=round(v_sol, 6),
                        v_tok=round(v_tok, 6),
                        buyer_pubkey=signer_pub,
                        data_lag_ms=round(data_lag, 2)
                    )
                    self._csv_write(EVENTS_CSV, row, EVENT_HEADER)

                except Exception as e:
                    self._log_error(st.mint, "WEBSOCKET_PARSE_ERROR", str(e),
                                    {"message": str(msg)[:500]})

    # ─────── enhanced RPC with better error handling ───────
    async def _rpc(self, payload, retries=3):
        """Enhanced RPC with exponential backoff and better error reporting"""
        last_error = None
        for attempt in range(retries):
            try:
                async with self.session.post(RPC, json=payload,
                                             timeout=aiohttp.ClientTimeout(total=CONFIG["rpc_timeout"])) as r:
                    if r.status >= 400:
                        raise aiohttp.ClientResponseError(
                            request_info=r.request_info,
                            history=r.history,
                            status=r.status
                        )

                    j = await r.json()
                    err = j.get("error")
                    if err:
                        raise RuntimeError(
                            f"RPC Error: {err.get('message', 'Unknown')}")
                    return j

            except Exception as e:
                last_error = e
                if attempt == retries - 1:
                    raise last_error

                # Exponential backoff
                delay = 0.5 * (2 ** attempt)
                await asyncio.sleep(delay)

        raise last_error

    # ─────── misc utils ───────
    def _csv_write(self, fn, row, header):
        """Thread-safe CSV writing with error handling"""
        try:
            with self.csv_lock:
                with open(fn, "a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=header).writerow(row)
        except Exception as e:
            self.log.error(f"Failed to write to {fn}: {e}")

    async def _monitor(self):
        """Enhanced monitoring with health checks"""
        last_health_check = time.time()
        while self.active:
            await asyncio.sleep(30)

            # Periodic health check
            if time.time() - last_health_check > 300:  # Every 5 minutes
                for mint, st in self.active.items():
                    try:
                        # Quick health check: can we fetch curve state?
                        await self._get_curve_validated(st.curve, mint)
                        self.log.debug(f"Health check passed for {mint}")
                    except Exception as e:
                        self._log_error(mint, "HEALTH_CHECK_FAILED", str(e))

                last_health_check = time.time()

        self.log.info("🎬 all samplers finished")

    async def _heartbeat(self):
        """Enhanced heartbeat with system stats"""
        while True:
            error_count = 0
            if os.path.exists(ERRORS_CSV):
                try:
                    with open(ERRORS_CSV, 'r') as f:
                        error_count = sum(1 for _ in f) - 1  # Subtract header
                except:
                    pass

            self.log.info(
                f"⏰ heartbeat | active={len(self.active)} | errors={error_count} | reconnects={self.reconnect_count}")
            await asyncio.sleep(300)


# ───────── main ─────────
if __name__ == "__main__":
    try:
        asyncio.run(Tracker().run())
    except KeyboardInterrupt:
        print("\n⏹ stopped")

```
