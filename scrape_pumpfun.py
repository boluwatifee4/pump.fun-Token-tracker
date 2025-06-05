#!/usr/bin/env python3
"""
pumpfun_v2_poll.py

Polls Bitquery V2's HTTP GraphQL endpoint to detect newly minted Pump.fun tokens
and stores them in CSV files for analysis.
"""

import os
import time
import json
import csv
from pathlib import Path
from datetime import datetime, timezone, timedelta

import requests
from dotenv import load_dotenv

# ─────────── Load environment variables ───────────
load_dotenv()
BITQUERY_OAUTH_TOKEN = os.getenv("BITQUERY_OAUTH_TOKEN")
if not BITQUERY_OAUTH_TOKEN:
    print("❌  Missing BITQUERY_OAUTH_TOKEN in .env. Please add your OAuth token.")
    exit(1)

# ─────────── Configuration ───────────
BITQUERY_URL = "https://streaming.bitquery.io/eap"
STATE_FILE = Path("bitquery_state.json")
POLL_INTERVAL = 30  # seconds
INITIAL_TIMEOUT = 60
NORMAL_TIMEOUT = 20

# CSV file paths
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

TOKENS_CSV = DATA_DIR / "pump_tokens.csv"
ALL_INSTRUCTIONS_CSV = DATA_DIR / "pump_instructions.csv"

# Headers for API requests
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {BITQUERY_OAUTH_TOKEN}"
}

# Working V2 GraphQL query for Solana
GRAPHQL_QUERY = """
query GetPumpFunTransactions($since: DateTime!) {
  Solana {
    Instructions(
      where: {
        Block: { Time: { after: $since } }
        Instruction: {
          Program: { Address: { is: "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P" } }
        }
      }
      orderBy: { ascending: Block_Time }
      limit: { count: 50 }
    ) {
      Block {
        Time
      }
      Transaction {
        Signature
        Signer
      }
      Instruction {
        Program {
          Address
          Method
        }
        Accounts {
          Address
          IsWritable
        }
      }
    }
  }
}
"""

def iso_now() -> str:
    """Return current UTC time as ISO8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def init_csv_files():
    """Initialize CSV files with headers if they don't exist."""
    
    # Tokens CSV - for new token discoveries (create instructions only)
    if not TOKENS_CSV.exists():
        with open(TOKENS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'discovered_at',
                'block_time',
                'transaction_signature',
                'signer',
                'method',
                'mint_address',
                'accounts_count',
                'all_accounts'
            ])
    
    # All instructions CSV - for complete pump.fun activity
    if not ALL_INSTRUCTIONS_CSV.exists():
        with open(ALL_INSTRUCTIONS_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'recorded_at',
                'block_time',
                'transaction_signature',
                'signer',
                'program_address',
                'method',
                'accounts_count',
                'first_writable_account',
                'all_accounts'
            ])

def save_token_discovery(token_data: dict):
    """Save a new token discovery to tokens CSV."""
    with open(TOKENS_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            token_data['discovered_at'],
            token_data['block_time'],
            token_data['transaction_signature'],
            token_data['signer'],
            token_data['method'],
            token_data['mint_address'],
            token_data['accounts_count'],
            token_data['all_accounts']
        ])

def save_instruction(instruction_data: dict):
    """Save instruction data to all instructions CSV."""
    with open(ALL_INSTRUCTIONS_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            instruction_data['recorded_at'],
            instruction_data['block_time'],
            instruction_data['transaction_signature'],
            instruction_data['signer'],
            instruction_data['program_address'],
            instruction_data['method'],
            instruction_data['accounts_count'],
            instruction_data['first_writable_account'],
            instruction_data['all_accounts']
        ])

def extract_account_info(accounts: list) -> tuple:
    """Extract mint address and format all accounts."""
    first_writable = None
    all_accounts = []
    
    for acc in accounts:
        account_info = f"{acc['Address']}({'W' if acc.get('IsWritable') else 'R'})"
        all_accounts.append(account_info)
        
        # First writable account is typically the mint
        if acc.get("IsWritable") and first_writable is None:
            first_writable = acc["Address"]
    
    return first_writable, "|".join(all_accounts)

def load_last_seen() -> str:
    """Load last seen timestamp from state file."""
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            ts = data.get("lastSeen")
            if isinstance(ts, str) and ts:
                return ts
        except Exception:
            pass
    
    one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
    return one_hour_ago.strftime("%Y-%m-%dT%H:%M:%SZ")

def save_last_seen(timestamp_iso: str):
    """Save last seen timestamp to state file."""
    STATE_FILE.write_text(json.dumps(
        {"lastSeen": timestamp_iso}), encoding="utf-8")

def fetch_new_tokens(since_ts: str, is_first_query: bool = False) -> list:
    """Fetch new pump.fun instructions from Bitquery."""
    payload = {
        "query": GRAPHQL_QUERY,
        "variables": {"since": since_ts}
    }
    
    timeout = INITIAL_TIMEOUT if is_first_query else NORMAL_TIMEOUT
    
    for attempt in range(3):
        try:
            print(f"[DEBUG] Attempt {attempt + 1}, timeout: {timeout}s")
            response = requests.post(
                BITQUERY_URL, headers=HEADERS, json=payload, timeout=timeout)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for GraphQL errors
            if "errors" in data:
                print(f"❌ GraphQL errors: {data['errors']}")
                return []
            
            # Navigate response structure
            solana_data = data.get("data", {}).get("Solana")
            if not solana_data:
                print("❌ No 'Solana' field in response")
                return []
                
            instructions = solana_data.get("Instructions", [])
            print(f"✅ Found {len(instructions)} instructions")
            return instructions
            
        except requests.exceptions.Timeout:
            print(f"⏰ Timeout on attempt {attempt + 1}")
            if attempt < 2:
                time.sleep(5)
            continue
        except Exception as e:
            print(f"❌ Error on attempt {attempt + 1}: {e}")
            if attempt < 2:
                time.sleep(5)
            continue
    
    print("❌ All attempts failed")
    return []

def main():
    # Initialize CSV files
    init_csv_files()
    print(f"📁 CSV files initialized in {DATA_DIR}/")
    print(f"📄 Tokens: {TOKENS_CSV}")
    print(f"📄 All Instructions: {ALL_INSTRUCTIONS_CSV}")
    
    last_seen = load_last_seen()
    print(f"\n[{iso_now()}] Starting pump.fun tracker; last_seen = {last_seen}")
    
    is_first_query = True
    discovered_tokens = set()  # Track already discovered tokens
    
    while True:
        try:
            instructions = fetch_new_tokens(last_seen, is_first_query)
            is_first_query = False
        except Exception as e:
            print(f"[{iso_now()}] ❌ Error fetching from Bitquery: {e}")
            time.sleep(POLL_INTERVAL)
            continue

        if instructions:
            current_time = iso_now()
            new_tokens_count = 0
            
            for entry in instructions:
                block_time = entry["Block"]["Time"]
                signature = entry["Transaction"]["Signature"]
                signer = entry["Transaction"]["Signer"]
                program_method = entry["Instruction"].get("Program", {}).get("Method", "unknown")
                program_address = entry["Instruction"]["Program"]["Address"]
                accounts = entry["Instruction"]["Accounts"]
                
                # Extract account information
                first_writable, all_accounts_str = extract_account_info(accounts)
                
                # Save all instructions to CSV
                instruction_data = {
                    'recorded_at': current_time,
                    'block_time': block_time,
                    'transaction_signature': signature,
                    'signer': signer,
                    'program_address': program_address,
                    'method': program_method,
                    'accounts_count': len(accounts),
                    'first_writable_account': first_writable or 'N/A',
                    'all_accounts': all_accounts_str
                }
                save_instruction(instruction_data)
                
                # Special handling for create instructions (new tokens)
                if program_method == "create" and first_writable:
                    if first_writable not in discovered_tokens:
                        discovered_tokens.add(first_writable)
                        new_tokens_count += 1
                        
                        token_data = {
                            'discovered_at': current_time,
                            'block_time': block_time,
                            'transaction_signature': signature,
                            'signer': signer,
                            'method': program_method,
                            'mint_address': first_writable,
                            'accounts_count': len(accounts),
                            'all_accounts': all_accounts_str
                        }
                        save_token_discovery(token_data)
                        
                        print(f"🚀 NEW TOKEN: {first_writable} by {signer[:8]}...")

            # Update last seen timestamp
            newest_time = instructions[-1]["Block"]["Time"]
            last_seen = newest_time
            save_last_seen(last_seen)
            
            print(f"💾 Saved {len(instructions)} instructions ({new_tokens_count} new tokens)")
        else:
            print(f"[{iso_now()}] ↻ No new instructions since {last_seen}")

        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
