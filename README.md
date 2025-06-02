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
