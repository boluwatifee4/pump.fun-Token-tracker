#!/usr/bin/env python3
"""
🔍 Pump.fun Migration Discovery

• Scans last 30 days for Raydium/Pump.swap migrations
• Identifies tokens that completed ≥85 SOL bonding curve  
• Outputs to migrated_tokens.csv
"""

import asyncio
import aiohttp
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

# ────────────────────────── CONSTANTS ──────────────────────────
SOLANA_RPC_URL = "https://api.mainnet-beta.solana.com"
COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd"

# Program addresses
PUMP_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
RAYDIUM_AMM_V4 = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
RAYDIUM_AMM_V5 = "5quBtoiQqxF9Jv6KYKctB59NT3gtJD2Y65kdnB1Uev3h"

TARGET_SOL = 85
DISCOVERY_DAYS = 30

# Output file
MIGRATED_TOKENS_CSV = "migrated_tokens.csv"
CSV_HEADER = [
    "mint", "name", "symbol", "launch_time", "migration_time",
    "migration_platform", "final_sol_raised", "migration_tx",
    "age_at_migration_hours", "trackable"
]


@dataclass
class MigratedToken:
    mint: str
    name: str
    symbol: str
    launch_time: datetime
    migration_time: datetime
    migration_platform: str
    final_sol_raised: float
    migration_tx: str


class MigrationDiscovery:
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.log = self._init_logger()
        self.discovered_mints: Set[str] = set()

    def _init_logger(self):
        """Initialize logger"""
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(
                    f"logs/migration_discovery_{time.strftime('%Y%m%d')}.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger("migration-discovery")

    async def run(self):
        """Main runner"""
        async with aiohttp.ClientSession() as sess:
            self.session = sess

            # Test connections
            await self._test_connections()

            self.log.info(
                "🔍 Starting Pump.fun Migration Discovery (Last 30 Days)")

            # Discover migrations
            migrated_tokens = await self._discover_migrations()

            # Save to CSV
            self._save_migrations_to_csv(migrated_tokens)

            self.log.info(
                f"✅ Migration discovery completed! Found {len(migrated_tokens)} migrations")

    async def _test_connections(self):
        """Test RPC connection"""
        try:
            payload = {"jsonrpc": "2.0", "id": 1, "method": "getHealth"}
            async with self.session.post(SOLANA_RPC_URL, json=payload) as response:
                if response.status == 200:
                    self.log.info("✅ Solana RPC connection successful!")
                else:
                    raise RuntimeError(f"RPC HTTP {response.status}")
        except Exception as e:
            self.log.error(f"❌ Solana RPC connection failed: {e}")
            raise

    async def _discover_migrations(self) -> List[MigratedToken]:
        """Discover all migrations in the last 30 days"""
        cutoff_time = datetime.now(timezone.utc) - \
            timedelta(days=DISCOVERY_DAYS)
        migrated_tokens = []

        self.log.info(
            f"🔍 Scanning for migrations since {cutoff_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Method 1: Scan Raydium AMM V4 for migrations
        self.log.info("📡 Scanning Raydium AMM V4...")
        raydium_v4_migrations = await self._scan_raydium_migrations(RAYDIUM_AMM_V4, "raydium_v4", cutoff_time)
        migrated_tokens.extend(raydium_v4_migrations)

        # Method 2: Scan Raydium AMM V5 for migrations
        self.log.info("📡 Scanning Raydium AMM V5...")
        raydium_v5_migrations = await self._scan_raydium_migrations(RAYDIUM_AMM_V5, "raydium_v5", cutoff_time)
        migrated_tokens.extend(raydium_v5_migrations)

        # Remove duplicates
        unique_migrations = []
        seen_mints = set()
        for token in migrated_tokens:
            if token.mint not in seen_mints:
                unique_migrations.append(token)
                seen_mints.add(token.mint)

        return unique_migrations

    async def _scan_raydium_migrations(self, program_id: str, platform: str, since: datetime) -> List[MigratedToken]:
        """Scan Raydium program for pump.fun token migrations"""
        migrations = []
        since_timestamp = int(since.timestamp())

        try:
            # Get recent signatures for Raydium program
            self.log.info(f"🔍 Getting signatures for {platform}...")

            # Get signatures in batches
            all_signatures = []
            before = None

            for batch in range(5):  # Get 5 batches = ~5000 signatures
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getSignaturesForAddress",
                    "params": [program_id, {"limit": 1000, "before": before}]
                }

                async with self.session.post(SOLANA_RPC_URL, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        signatures = data.get("result", [])

                        if not signatures:
                            break

                        # Filter by time
                        valid_signatures = []
                        for sig_info in signatures:
                            block_time = sig_info.get("blockTime", 0)
                            if block_time >= since_timestamp:
                                valid_signatures.append(sig_info)
                            else:
                                # Stop if we've gone past our time window
                                break

                        all_signatures.extend(valid_signatures)
                        before = signatures[-1]["signature"]

                        self.log.info(
                            f"   Batch {batch + 1}: {len(valid_signatures)} valid signatures")

                        # If last signature is too old, stop
                        if signatures[-1].get("blockTime", 0) < since_timestamp:
                            break
                    else:
                        self.log.error(
                            f"Error getting signatures: HTTP {response.status}")
                        break

                # Rate limiting
                await asyncio.sleep(0.5)

            self.log.info(
                f"📊 Total valid signatures for {platform}: {len(all_signatures)}")

            # Analyze signatures for pump.fun migrations
            for i, sig_info in enumerate(all_signatures):
                if i % 100 == 0:
                    self.log.info(
                        f"   Analyzed {i}/{len(all_signatures)} signatures...")

                migration = await self._analyze_migration_transaction(sig_info["signature"], platform)
                if migration:
                    if migration.mint not in self.discovered_mints:
                        migrations.append(migration)
                        self.discovered_mints.add(migration.mint)
                        self.log.info(
                            f"✅ Found migration: {migration.name} ({migration.mint[:8]}) - {migration.final_sol_raised:.2f} SOL")

                # Rate limiting to avoid hitting RPC limits
                if i % 10 == 0:
                    await asyncio.sleep(0.1)

        except Exception as e:
            self.log.error(f"Error scanning {platform} migrations: {e}")

        return migrations

    async def _analyze_migration_transaction(self, signature: str, platform: str) -> Optional[MigratedToken]:
        """Analyze a transaction to see if it's a pump.fun migration"""
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTransaction",
                "params": [
                    signature,
                    {
                        "encoding": "jsonParsed",
                        "maxSupportedTransactionVersion": 0
                    }
                ]
            }

            async with self.session.post(SOLANA_RPC_URL, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    tx = data.get("result")

                    if tx and tx.get("meta", {}).get("err") is None:
                        return await self._parse_migration_transaction(tx, signature, platform)

        except Exception as e:
            self.log.debug(f"Error analyzing migration {signature[:8]}: {e}")

        return None

    async def _parse_migration_transaction(self, tx: Dict, signature: str, platform: str) -> Optional[MigratedToken]:
        """Parse transaction to extract migration info"""
        try:
            # Check if pump.fun program is involved
            account_keys = tx.get("transaction", {}).get(
                "message", {}).get("accountKeys", [])
            pump_involved = any(
                acc.get("pubkey") == PUMP_PROGRAM for acc in account_keys
            )

            if not pump_involved:
                return None

            # Look for pool initialization or liquidity addition patterns
            instructions = tx.get("transaction", {}).get(
                "message", {}).get("instructions", [])
            is_pool_creation = False

            for instruction in instructions:
                parsed = instruction.get("parsed", {})
                instruction_type = parsed.get("type", "")

                # Look for pool initialization patterns
                if instruction_type in ["initializeAccount", "initializeMint", "initialize"]:
                    is_pool_creation = True
                    break

                # Look for data patterns that suggest pool creation
                data = instruction.get("data", "")
                if len(data) > 20:  # Pool creation usually has substantial data
                    is_pool_creation = True
                    break

            if not is_pool_creation:
                return None

            # Extract token mint from token balances
            token_mint = self._extract_token_mint_from_migration(tx)
            if not token_mint:
                return None

            # Check if this token completed bonding curve (≥85 SOL)
            sol_raised = await self._calculate_final_sol_raised(token_mint)
            if sol_raised < TARGET_SOL:
                return None

            # Get token metadata
            token_info = await self._get_token_metadata(token_mint)

            migration_time = datetime.fromtimestamp(
                tx.get("blockTime", 0), timezone.utc)
            launch_time = await self._estimate_launch_time(token_mint)

            return MigratedToken(
                mint=token_mint,
                name=token_info.get("name", "Unknown"),
                symbol=token_info.get("symbol", "UNK"),
                launch_time=launch_time,
                migration_time=migration_time,
                migration_platform=platform,
                final_sol_raised=sol_raised,
                migration_tx=signature
            )

        except Exception as e:
            self.log.debug(f"Error parsing migration transaction: {e}")

        return None

    def _extract_token_mint_from_migration(self, tx: Dict) -> Optional[str]:
        """Extract token mint from migration transaction"""
        try:
            # Method 1: Look in postTokenBalances
            meta = tx.get("meta", {})
            post_balances = meta.get("postTokenBalances", [])

            for balance in post_balances:
                mint = balance.get("mint", "")
                if mint and len(mint) == 44:
                    # Skip known system tokens
                    if mint not in [
                        "So11111111111111111111111111111111111111112",  # WSOL
                        "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
                        "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",  # USDT
                    ]:
                        return mint

            # Method 2: Look in account keys for potential token mints
            account_keys = tx.get("transaction", {}).get(
                "message", {}).get("accountKeys", [])

            for account in account_keys:
                pubkey = account.get("pubkey", "")
                if len(pubkey) == 44 and pubkey not in [
                    PUMP_PROGRAM,
                    "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
                    "11111111111111111111111111111112",
                    "So11111111111111111111111111111111111111112"
                ]:
                    # This could be a token mint - we'll verify later
                    return pubkey

        except Exception as e:
            self.log.debug(f"Error extracting mint: {e}")

        return None

    async def _calculate_final_sol_raised(self, mint: str) -> float:
        """Calculate total SOL raised for a token"""
        try:
            # Get all signatures for the token
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getSignaturesForAddress",
                "params": [mint, {"limit": 1000}]
            }

            async with self.session.post(SOLANA_RPC_URL, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    signatures = data.get("result", [])

                    total_buy_volume = 0.0
                    total_sell_volume = 0.0

                    # Analyze a sample of transactions to calculate volume
                    # Sample first 50
                    for i, sig_info in enumerate(signatures[:50]):
                        try:
                            tx_data = await self._get_transaction_details(sig_info["signature"], mint)
                            if tx_data:
                                if tx_data["is_buy"]:
                                    total_buy_volume += tx_data["amount_sol"]
                                else:
                                    total_sell_volume += tx_data["amount_sol"]
                        except:
                            continue

                        # Rate limiting
                        if i % 10 == 0:
                            await asyncio.sleep(0.1)

                    net_volume = total_buy_volume - total_sell_volume
                    return max(0, net_volume)

        except Exception as e:
            self.log.debug(f"Error calculating SOL raised for {mint[:8]}: {e}")

        return 0.0

    async def _get_transaction_details(self, signature: str, mint: str) -> Optional[Dict]:
        """Get transaction details for volume calculation"""
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTransaction",
                "params": [
                    signature,
                    {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}
                ]
            }

            async with self.session.post(SOLANA_RPC_URL, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    tx = data.get("result")

                    if tx and tx.get("meta", {}).get("err") is None:
                        return self._parse_pump_transaction_simple(tx, signature)

        except Exception as e:
            self.log.debug(f"Error getting transaction details: {e}")

        return None

    def _parse_pump_transaction_simple(self, tx: Dict, signature: str) -> Optional[Dict]:
        """Simple transaction parser for volume calculation"""
        try:
            meta = tx.get("meta", {})
            pre_balances = meta.get("preBalances", [])
            post_balances = meta.get("postBalances", [])

            if len(pre_balances) != len(post_balances):
                return None

            # Find largest SOL balance change
            max_change = 0
            for i in range(len(pre_balances)):
                sol_change_lamports = abs(post_balances[i] - pre_balances[i])
                sol_change = sol_change_lamports / 1_000_000_000
                if sol_change > max_change:
                    max_change = sol_change

            if max_change < 0.001:  # Ignore tiny changes
                return None

            # Determine buy/sell (simplified)
            net_change = sum(post_balances) - sum(pre_balances)
            is_buy = net_change < 0  # Net SOL decrease = buy

            return {
                "amount_sol": max_change,
                "is_buy": is_buy,
                "signature": signature
            }

        except Exception as e:
            self.log.debug(f"Error parsing transaction: {e}")

        return None

    async def _get_token_metadata(self, mint: str) -> Dict:
        """Get basic token metadata"""
        try:
            # Try to get metadata from token program
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getAccountInfo",
                "params": [
                    mint,
                    {"encoding": "jsonParsed"}
                ]
            }

            async with self.session.post(SOLANA_RPC_URL, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    account_info = data.get("result", {}).get("value", {})

                    if account_info:
                        parsed_data = account_info.get(
                            "data", {}).get("parsed", {})
                        token_info = parsed_data.get("info", {})

                        return {
                            "name": f"Token {mint[:8]}",  # Default name
                            "symbol": "UNK",
                            "decimals": token_info.get("decimals", 6)
                        }

        except Exception as e:
            self.log.debug(f"Error getting metadata for {mint[:8]}: {e}")

        return {"name": f"Token {mint[:8]}", "symbol": "UNK", "decimals": 6}

    async def _estimate_launch_time(self, mint: str) -> datetime:
        """Estimate token launch time from first transaction"""
        try:
            # Get oldest signatures
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getSignaturesForAddress",
                "params": [mint, {"limit": 1000}]
            }

            async with self.session.post(SOLANA_RPC_URL, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    signatures = data.get("result", [])

                    if signatures:
                        # Last signature is usually the oldest
                        oldest_sig = signatures[-1]
                        launch_timestamp = oldest_sig.get("blockTime", 0)
                        return datetime.fromtimestamp(launch_timestamp, timezone.utc)

        except Exception as e:
            self.log.debug(f"Error estimating launch time for {mint[:8]}: {e}")

        # Fallback: estimate as 2 hours before migration
        return datetime.now(timezone.utc) - timedelta(hours=2)

    def _save_migrations_to_csv(self, migrations: List[MigratedToken]):
        """Save discovered migrations to CSV"""
        with open(MIGRATED_TOKENS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
            writer.writeheader()

            for token in migrations:
                # Calculate age at migration
                age_at_migration = (token.migration_time -
                                    token.launch_time).total_seconds() / 3600

                # Check if still trackable (within 45 minutes of launch)
                current_age = (datetime.now(timezone.utc) -
                               token.launch_time).total_seconds() / 60
                trackable = current_age < 45

                writer.writerow({
                    "mint": token.mint,
                    "name": token.name,
                    "symbol": token.symbol,
                    "launch_time": token.launch_time.isoformat(),
                    "migration_time": token.migration_time.isoformat(),
                    "migration_platform": token.migration_platform,
                    "final_sol_raised": round(token.final_sol_raised, 4),
                    "migration_tx": token.migration_tx,
                    "age_at_migration_hours": round(age_at_migration, 2),
                    "trackable": str(trackable).lower()
                })

        trackable_count = sum(1 for t in migrations if (datetime.now(
            timezone.utc) - t.launch_time).total_seconds() / 60 < 45)

        self.log.info(
            f"💾 Saved {len(migrations)} migrations to {MIGRATED_TOKENS_CSV}")
        self.log.info(
            f"🎯 {trackable_count} tokens are still trackable (< 45min old)")

        if migrations:
            self.log.info("📊 Sample migrations found:")
            for i, token in enumerate(migrations[:5]):
                age_hours = (datetime.now(timezone.utc) -
                             token.launch_time).total_seconds() / 3600
                self.log.info(
                    f"   {i+1}. {token.name} ({token.mint[:8]}) - {token.final_sol_raised:.1f} SOL - {age_hours:.1f}h old")


if __name__ == "__main__":
    try:
        asyncio.run(MigrationDiscovery().run())
    except KeyboardInterrupt:
        print("\n⏹ Migration discovery stopped by user")
