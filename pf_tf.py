#!/usr/bin/env python3
"""
📋 Pump.fun Token Fetcher - Moralis Edition

• Fetches newly launched pump.fun tokens from Moralis
• Saves them to CSV for processing by tracker script
• Run once to get list of tokens to track
"""

import asyncio
import aiohttp
import csv
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List

# ────────────────────────── constants ──────────────────────────
MORALIS_API_KEY = ""  # Replace with your Moralis key
MORALIS_BASE_URL = "https://solana-gateway.moralis.io"

TOKEN_LIST_CSV = "pump_tokens.csv"
TOKEN_CSV_HEADER = [
    "tokenAddress", "name", "symbol", "logo", "decimals",
    "priceNative", "priceUsd", "liquidity", "fullyDilutedValuation",
    "createdAt", "age_minutes", "trackable"
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TokenFetcher:
    def __init__(self) -> None:
        self.session = None
        self.log = self._init_logger()

    async def fetch_and_save_tokens(self, minutes_back: int = 60, limit: int = 100):
        """Fetch recent tokens and save to CSV"""
        async with aiohttp.ClientSession() as sess:
            self.session = sess

            self.log.info(
                f"🔍 Fetching pump.fun tokens from last {minutes_back} minutes...")

            # Test connection
            await self._test_connection()

            # Fetch tokens
            tokens = await self._fetch_recent_tokens(limit)

            if not tokens:
                self.log.warning("❌ No tokens fetched from Moralis")
                return

            # Filter and process tokens
            recent_tokens = self._filter_recent_tokens(tokens, minutes_back)

            # Save to CSV
            saved_count = self._save_tokens_to_csv(recent_tokens)

            self.log.info(
                f"✅ Saved {saved_count} trackable tokens to {TOKEN_LIST_CSV}")
            self.log.info(f"🚀 Ready to run tracker script!")

    async def _test_connection(self):
        """Test Moralis API connection"""
        url = f"{MORALIS_BASE_URL}/token/mainnet/exchange/pumpfun/new?limit=1"
        headers = {"X-API-Key": MORALIS_API_KEY, "accept": "application/json"}

        try:
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    self.log.info("✅ Moralis API connection successful!")
                    return True
                else:
                    raise RuntimeError(f"HTTP {response.status}")
        except Exception as e:
            self.log.error(f"❌ Moralis API connection failed: {e}")
            raise

    async def _fetch_recent_tokens(self, limit: int) -> List[Dict]:
        """Fetch tokens from Moralis"""
        url = f"{MORALIS_BASE_URL}/token/mainnet/exchange/pumpfun/new?limit={limit}"
        headers = {"X-API-Key": MORALIS_API_KEY, "accept": "application/json"}

        try:
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    tokens = data.get("result", [])
                    self.log.info(
                        f"📦 Fetched {len(tokens)} tokens from Moralis")
                    return tokens
                else:
                    text = await response.text()
                    self.log.error(
                        f"Moralis API error {response.status}: {text}")
                    return []
        except Exception as e:
            self.log.error(f"Error fetching tokens: {e}")
            return []

    def _filter_recent_tokens(self, tokens: List[Dict], minutes_back: int) -> List[Dict]:
        """Filter tokens to only recent ones and add metadata"""
        cutoff_time = datetime.now(timezone.utc) - \
            timedelta(minutes=minutes_back)
        recent_tokens = []

        for token in tokens:
            try:
                created_at = datetime.fromisoformat(
                    token["createdAt"].replace("Z", "+00:00"))
                age_minutes = (datetime.now(timezone.utc) -
                               created_at).total_seconds() / 60

                # Add calculated fields
                token["age_minutes"] = round(age_minutes, 1)
                # Still trackable if less than 45min old
                token["trackable"] = age_minutes < 45

                if created_at >= cutoff_time:
                    recent_tokens.append(token)

            except Exception as e:
                self.log.debug(
                    f"Error processing token {token.get('tokenAddress', 'unknown')}: {e}")
                continue

        self.log.info(
            f"🎯 Filtered to {len(recent_tokens)} tokens within {minutes_back}min window")
        trackable_count = sum(1 for t in recent_tokens if t["trackable"])
        self.log.info(
            f"📡 {trackable_count} tokens are still trackable (< 45min old)")

        return recent_tokens

    def _save_tokens_to_csv(self, tokens: List[Dict]) -> int:
        """Save tokens to CSV file with backup system"""
        if not tokens:
            return 0

        # Step 1: Backup existing tokens before overwriting
        if os.path.exists(TOKEN_LIST_CSV):
            self.log.info(f"📦 Backing up existing tokens to backlog...")
            self._backup_existing_tokens()

        # Step 2: Process new tokens
        processed_tokens = []
        for token in tokens:
            processed_token = {}
            for header in TOKEN_CSV_HEADER:
                processed_token[header] = token.get(header, "")
            processed_tokens.append(processed_token)

        # Step 3: Write new tokens to main CSV
        with open(TOKEN_LIST_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TOKEN_CSV_HEADER)
            writer.writeheader()
            writer.writerows(processed_tokens)

        # Step 4: Also append new tokens to backlog
        self._append_tokens_to_backlog(processed_tokens)

        # Log summary
        trackable_tokens = [t for t in processed_tokens if t["trackable"]]

        self.log.info(f"📄 CSV saved with {len(processed_tokens)} total tokens")
        self.log.info(f"🎯 {len(trackable_tokens)} tokens ready for tracking")
        self.log.info(f"📚 Updated backlog with new tokens")

        if trackable_tokens:
            self.log.info("🚀 Next trackable tokens:")
            for i, token in enumerate(trackable_tokens[:5]):  # Show first 5
                self.log.info(
                    f"   {i+1}. {token['name']} ({token['tokenAddress'][:8]}) - {token['age_minutes']}min old")
            if len(trackable_tokens) > 5:
                self.log.info(f"   ... and {len(trackable_tokens) - 5} more")

        return len(trackable_tokens)

    def _backup_existing_tokens(self):
        """Move existing tokens from pump_tokens.csv to backlog"""
        backlog_file = "backlog_pump_tokens.csv"

        try:
            # Read existing tokens from pump_tokens.csv
            existing_tokens = []
            with open(TOKEN_LIST_CSV, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_tokens = list(reader)

            if existing_tokens:
                self.log.info(
                    f"   Found {len(existing_tokens)} existing tokens to backup")

                # Read current backlog if it exists
                backlog_tokens = []
                if os.path.exists(backlog_file):
                    with open(backlog_file, 'r', newline='', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        backlog_tokens = list(reader)

                # Get existing token addresses to avoid duplicates
                existing_addresses = {
                    token.get('tokenAddress', '') for token in backlog_tokens}

                # Add new tokens (avoid duplicates)
                new_tokens_added = 0
                for token in existing_tokens:
                    token_address = token.get('tokenAddress', '')
                    if token_address and token_address not in existing_addresses:
                        backlog_tokens.append(token)
                        existing_addresses.add(token_address)
                        new_tokens_added += 1

                # Write updated backlog
                self._write_tokens_to_file(backlog_tokens, backlog_file)
                self.log.info(
                    f"   Added {new_tokens_added} new tokens to backlog")

        except Exception as e:
            self.log.warning(f"   ⚠️  Error backing up tokens: {e}")

    def _append_tokens_to_backlog(self, new_tokens: List[Dict]):
        """Append new tokens to backlog (avoid duplicates)"""
        backlog_file = "backlog_pump_tokens.csv"

        try:
            # Read current backlog
            backlog_tokens = []
            if os.path.exists(backlog_file):
                with open(backlog_file, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    backlog_tokens = list(reader)

            # Get existing addresses
            existing_addresses = {
                token.get('tokenAddress', '') for token in backlog_tokens}

            # Add new unique tokens
            new_tokens_added = 0
            for token in new_tokens:
                token_address = token.get('tokenAddress', '')
                if token_address and token_address not in existing_addresses:
                    backlog_tokens.append(token)
                    existing_addresses.add(token_address)
                    new_tokens_added += 1

            # Write updated backlog
            if new_tokens_added > 0:
                self._write_tokens_to_file(backlog_tokens, backlog_file)
                self.log.info(
                    f"   📚 Added {new_tokens_added} new tokens to backlog")

        except Exception as e:
            self.log.warning(f"   ⚠️  Error updating backlog: {e}")

    def _write_tokens_to_file(self, tokens: List[Dict], filename: str):
        """Write tokens to specified CSV file"""
        with open(filename, "w", newline="", encoding="utf-8") as f:
            if tokens:
                # Use the fieldnames from the first token, or fallback to default
                fieldnames = list(
                    tokens[0].keys()) if tokens else TOKEN_CSV_HEADER
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(tokens)

    def show_backlog_stats(self):
        """Show statistics about the token backlog"""
        backlog_file = "backlog_pump_tokens.csv"

        if os.path.exists(backlog_file):
            try:
                with open(backlog_file, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    backlog_tokens = list(reader)

                total_tokens = len(backlog_tokens)

                # Count trackable tokens
                trackable_tokens = [t for t in backlog_tokens if t.get(
                    'trackable', '').lower() == 'true']

                self.log.info(f"📚 Backlog Statistics:")
                self.log.info(f"   Total tokens in backlog: {total_tokens}")
                self.log.info(f"   Still trackable: {len(trackable_tokens)}")
                self.log.info(f"   File: {backlog_file}")

            except Exception as e:
                self.log.warning(f"Error reading backlog: {e}")
        else:
            self.log.info("📚 No backlog file exists yet")

    def _init_logger(self):
        """Initialize logger"""
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(
                    f"logs/token_fetcher_{time.strftime('%Y%m%d')}.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger("token-fetcher")


# ────────────────────────── bootstrap ────────────────────────
if __name__ == "__main__":
    import argparse

    # Check for API key
    if MORALIS_API_KEY == "YOUR_MORALIS_API_KEY":
        print("❌ Please set your MORALIS_API_KEY in the script")
        sys.exit(1)

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Fetch pump.fun tokens from Moralis")
    parser.add_argument("--minutes", type=int, default=60,
                        help="How many minutes back to fetch tokens (default: 60)")
    parser.add_argument("--limit", type=int, default=100,
                        help="Maximum tokens to fetch from API (default: 100)")

    args = parser.parse_args()

    async def main():
        fetcher = TokenFetcher()
        await fetcher.fetch_and_save_tokens(args.minutes, args.limit)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹ Stopped by user")
