#!/usr/bin/env python3
"""
🔥 Fresh Token Fetcher for Pump.fun
Gets recently launched tokens WITHOUT activity filtering
"""

import asyncio
import aiohttp
import csv
import json
from datetime import datetime, timezone
import ssl
import os  # Add this line with other imports

# Updated API endpoint (pump.fun changed their API)
PUMP_API_URL = "https://frontend-api.pump.fun/coins"
SOLANA_RPC_URL = "https://api.mainnet-beta.solana.com"


async def get_fresh_active_tokens():
    """Get fresh tokens WITHOUT activity filtering"""
    # Create SSL context and connector with better settings
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    connector = aiohttp.TCPConnector(
        ssl=ssl_context,
        ttl_dns_cache=300,
        use_dns_cache=True,
        limit=100,
        limit_per_host=10
    )

    timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_read=10)

    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
    ) as session:

        print("🔍 Fetching fresh tokens from pump.fun...")

        tokens = []

        # Method 1: Try the official API
        api_tokens = await try_pump_api(session)
        if api_tokens:
            tokens = api_tokens[:30]
            print(f"✅ Got {len(tokens)} tokens from API")

        # Method 2: Extract from transactions (improved)
        if not tokens:
            print("🔄 Trying alternative method...")
            tokens = await get_tokens_alternative(session)

        if tokens:
            print(f"✅ Found {len(tokens)} fresh tokens")
            save_to_csv(tokens)
            return tokens
        else:
            print("❌ No tokens found - pump.fun API might be down")
            # Create a minimal CSV with some test tokens for now
            create_test_csv()
            return []


def create_test_csv():
    """Create a test CSV with some known tokens for testing"""
    test_tokens = [
        {
            "mint": "2DyxvdkEM1c3D9FZk3nUHLd1NkrisiLbo1dFQC62pump",
            "name": "Test Token 1",
            "symbol": "TEST1",
            "created_timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
            "usd_market_cap": 1000
        },
        {
            "mint": "Ga5snP518uXRdX1621Efk3HKDMfcWpDBvm6KBJG17rwA",
            "name": "Test Token 2",
            "symbol": "TEST2",
            "created_timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
            "usd_market_cap": 2000
        }
    ]

    save_to_csv(test_tokens)
    print("📄 Created test CSV with 2 tokens for testing")


async def try_pump_api(session):
    """Try to get tokens from pump.fun API"""
    try:
        params = {
            "sort": "created_timestamp",  # Sort by newest first
            "order": "DESC",
            "limit": 50
        }

        print(f"📡 Trying: {PUMP_API_URL}")
        async with session.get(PUMP_API_URL, params=params) as response:
            print(f"Response status: {response.status}")

            if response.status == 200:
                text = await response.text()
                print(f"Response length: {len(text)} characters")

                try:
                    tokens = json.loads(text)
                    print(f"📦 Found {len(tokens)} tokens from API")
                    return tokens
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                    print(f"Response preview: {text[:200]}...")
                    return None
            else:
                text = await response.text()
                print(f"❌ API returned {response.status}: {text[:200]}")
                return None

    except Exception as e:
        print(f"❌ Error fetching from pump.fun API: {e}")
        return None


async def get_tokens_alternative(session):
    """Alternative method: Get tokens from recent Solana program activity"""
    print("🔍 Getting tokens from recent Solana activity...")

    try:
        # Get recent pump.fun program signatures
        pump_program = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getSignaturesForAddress",
            "params": [pump_program, {"limit": 100}]
        }

        async with session.post(SOLANA_RPC_URL, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                signatures = data.get("result", [])

                print(
                    f"📡 Found {len(signatures)} recent pump.fun transactions")

                # Extract unique token addresses from recent transactions
                unique_tokens = await extract_tokens_from_signatures(session, signatures[:30])

                return unique_tokens[:20]  # Return top 20

    except Exception as e:
        print(f"Error with alternative method: {e}")
        return []


async def extract_tokens_from_signatures(session, signatures):
    """Extract token addresses from transaction signatures - WITH DEBUGGING"""
    tokens = []
    seen_mints = set()

    print(f"🔍 Processing {len(signatures)} signatures...")

    for i, sig_info in enumerate(signatures):
        try:
            signature = sig_info["signature"]

            # Show progress
            if i % 5 == 0:
                print(f"   Processing signature {i+1}/{len(signatures)}")

            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTransaction",
                "params": [signature, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}]
            }

            async with session.post(SOLANA_RPC_URL, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    tx = data.get("result")

                    if tx and tx.get("meta", {}).get("err") is None:
                        # DEBUG: Print transaction structure
                        print(f"\n🔍 Transaction {i+1} structure:")
                        print(f"   Signature: {signature[:16]}...")

                        # Check account keys
                        account_keys = tx.get("transaction", {}).get(
                            "message", {}).get("accountKeys", [])
                        print(f"   Account keys: {len(account_keys)}")
                        # Show first 5
                        for j, acc in enumerate(account_keys[:5]):
                            print(f"     {j}: {acc.get('pubkey', '')[:8]}...")

                        # Check instructions
                        instructions = tx.get("transaction", {}).get(
                            "message", {}).get("instructions", [])
                        print(f"   Instructions: {len(instructions)}")
                        for j, inst in enumerate(instructions):
                            program_id = inst.get("programId", "")
                            print(f"     {j}: Program {program_id[:8]}...")
                            if program_id == "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P":
                                print(f"        ✅ PUMP.FUN INSTRUCTION FOUND!")
                                accounts = inst.get("accounts", [])
                                print(f"        Accounts: {len(accounts)}")
                                for k, acc in enumerate(accounts[:3]):
                                    print(f"          {k}: {acc[:8]}...")

                        # Check token balances
                        meta = tx.get("meta", {})
                        pre_balances = meta.get("preTokenBalances", [])
                        post_balances = meta.get("postTokenBalances", [])
                        print(
                            f"   Token balances: pre={len(pre_balances)}, post={len(post_balances)}")

                        for j, balance in enumerate(post_balances[:3]):
                            mint = balance.get("mint", "")
                            print(f"     {j}: Mint {mint[:8]}...")

                        # Extract token mint from transaction
                        mint = extract_mint_from_transaction(tx)
                        if mint:
                            print(f"🎯 EXTRACTED MINT: {mint}")

                            if mint not in seen_mints and len(mint) == 44:
                                seen_mints.add(mint)
                                print(f"🔍 Verifying token: {mint[:8]}...")

                                # Simplified verification - just check if it looks like a token
                                tokens.append({
                                    "mint": mint,
                                    "name": f"Token {mint[:8]}",
                                    "symbol": "UNK",
                                    "created_timestamp": sig_info.get("blockTime", 0) * 1000,
                                    "usd_market_cap": 0
                                })

                                print(f"✅ Added token: {mint[:8]}")

                                # Stop when we have enough tokens
                                if len(tokens) >= 5:  # Lower threshold for testing
                                    break
                            else:
                                print(f"🔄 Duplicate or invalid: {mint[:8]}")
                        else:
                            print(f"❌ No mint extracted from transaction")
                    else:
                        print(f"❌ Transaction failed or null")
                else:
                    print(f"❌ RPC error: {response.status}")

        except Exception as e:
            print(f"❌ Error processing signature {i}: {e}")
            continue

        # Small delay to avoid rate limits
        await asyncio.sleep(0.2)

    print(f"\n✅ Final result: {len(tokens)} unique tokens extracted")
    return tokens


# ALSO REPLACE the extract_mint_from_transaction function with this simpler version:
def extract_mint_from_transaction(tx):
    """Extract token mint address from transaction data - SIMPLIFIED"""
    try:
        # Method 1: Look in postTokenBalances (most reliable)
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
                    print(f"   🎯 Found mint in token balances: {mint[:8]}...")
                    return mint

        # Method 2: Look in pump.fun instruction accounts
        instructions = tx.get("transaction", {}).get(
            "message", {}).get("instructions", [])

        for instruction in instructions:
            program_id = instruction.get("programId", "")
            if program_id == "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P":
                accounts = instruction.get("accounts", [])
                # In pump.fun, the token mint is often the 2nd or 3rd account
                # Check positions 1, 2, 3
                for i, account in enumerate(accounts[1:4]):
                    if len(account) == 44 and account not in [
                        "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P",
                        "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
                        "11111111111111111111111111111112",
                        "So11111111111111111111111111111111111111112"
                    ]:
                        print(
                            f"   🎯 Found mint in instruction[{i+1}]: {account[:8]}...")
                        return account

        # Method 3: Look through all unique account keys
        account_keys = tx.get("transaction", {}).get(
            "message", {}).get("accountKeys", [])

        unique_accounts = set()
        for account in account_keys:
            pubkey = account.get("pubkey", "")
            if len(pubkey) == 44:
                unique_accounts.add(pubkey)

        # Filter out known system accounts
        system_accounts = {
            "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P",  # Pump program
            "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",  # Token program
            "11111111111111111111111111111112",  # System program
            "So11111111111111111111111111111111111111112",   # WSOL
            "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL",  # Associated token
            "SysvarRent111111111111111111111111111111111",    # Sysvar rent
            "ComputeBudget111111111111111111111111111111",   # Compute budget
        }

        potential_mints = unique_accounts - system_accounts

        if potential_mints:
            # Return the first potential mint
            mint = list(potential_mints)[0]
            print(f"   🎯 Found potential mint in accounts: {mint[:8]}...")
            return mint

    except Exception as e:
        print(f"   ❌ Error extracting mint: {e}")

    return None


def save_to_csv(tokens):
    """Save fresh tokens to CSV with backup system"""
    filename = "pump_tokens.csv"
    backlog_filename = "backlog_pump_tokens.csv"

    # Step 1: If pump_tokens.csv exists, backup/merge with backlog
    if os.path.exists(filename):
        print(f"📦 Backing up existing tokens to {backlog_filename}...")
        backup_existing_tokens(filename, backlog_filename)

    # Step 2: Write new tokens to pump_tokens.csv
    write_tokens_to_csv(tokens, filename)

    # Step 3: Also append new tokens to backlog
    append_tokens_to_backlog(tokens, backlog_filename)

    print(f"💾 Saved {len(tokens)} fresh tokens to {filename}")
    print(f"📚 Updated backlog with new tokens in {backlog_filename}")


def backup_existing_tokens(source_file, backlog_file):
    """Move existing tokens from source to backlog"""
    try:
        # Read existing tokens from pump_tokens.csv
        existing_tokens = []
        with open(source_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_tokens = list(reader)

        if existing_tokens:
            print(f"   Found {len(existing_tokens)} existing tokens to backup")

            # Read current backlog if it exists
            backlog_tokens = []
            if os.path.exists(backlog_file):
                with open(backlog_file, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    backlog_tokens = list(reader)

            # Get existing token addresses to avoid duplicates
            existing_addresses = {token.get('tokenAddress', '') for token in backlog_tokens}

            # Add new tokens (avoid duplicates)
            new_tokens_added = 0
            for token in existing_tokens:
                token_address = token.get('tokenAddress', '')
                if token_address and token_address not in existing_addresses:
                    backlog_tokens.append(token)
                    existing_addresses.add(token_address)
                    new_tokens_added += 1

            # Write updated backlog
            write_tokens_to_csv(backlog_tokens, backlog_file)
            print(f"   Added {new_tokens_added} new tokens to backlog")

    except Exception as e:
        print(f"   ⚠️  Error backing up tokens: {e}")


def append_tokens_to_backlog(new_tokens, backlog_file):
    """Append new tokens to backlog (avoid duplicates)"""
    try:
        # Read current backlog
        backlog_tokens = []
        if os.path.exists(backlog_file):
            with open(backlog_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                backlog_tokens = list(reader)

        # Get existing addresses
        existing_addresses = {token.get('tokenAddress', '') for token in backlog_tokens}

        # Add new unique tokens
        new_tokens_added = 0
        for token in new_tokens:
            token_address = token.get('mint', '')
            if token_address and token_address not in existing_addresses:
                # Convert token format for backlog
                backlog_token = {
                    'tokenAddress': token.get('mint', ''),
                    'name': token.get('name', 'Unknown'),
                    'symbol': token.get('symbol', 'UNK'),
                    'createdAt': calculate_created_at(token),
                    'age_minutes': calculate_age_minutes(token),
                    'trackable': str(calculate_age_minutes(token) < 45).lower(),
                    'market_cap_usd': token.get('usd_market_cap', 0)
                }
                backlog_tokens.append(backlog_token)
                existing_addresses.add(token_address)
                new_tokens_added += 1

        # Write updated backlog
        if new_tokens_added > 0:
            write_tokens_to_csv(backlog_tokens, backlog_file)
            print(f"   📚 Added {new_tokens_added} new tokens to backlog")

    except Exception as e:
        print(f"   ⚠️  Error updating backlog: {e}")


def write_tokens_to_csv(tokens, filename):
    """Write tokens to CSV file"""
    fieldnames = [
        "tokenAddress", "name", "symbol", "createdAt",
        "age_minutes", "trackable", "market_cap_usd"
    ]

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for token in tokens:
            # Handle both formats (new tokens and existing backlog tokens)
            if 'mint' in token:  # New token format
                writer.writerow({
                    "tokenAddress": token.get("mint", ""),
                    "name": token.get("name", "Unknown"),
                    "symbol": token.get("symbol", "UNK"),
                    "createdAt": calculate_created_at(token),
                    "age_minutes": calculate_age_minutes(token),
                    "trackable": str(calculate_age_minutes(token) < 45).lower(),
                    "market_cap_usd": token.get("usd_market_cap", 0)
                })
            else:  # Existing backlog format
                writer.writerow(token)


def calculate_created_at(token):
    """Calculate created_at timestamp"""
    created_at = token.get("created_timestamp", 0)
    if created_at:
        if created_at > 1000000000000:  # Timestamp in milliseconds
            created_dt = datetime.fromtimestamp(created_at / 1000, timezone.utc)
        else:  # Timestamp in seconds
            created_dt = datetime.fromtimestamp(created_at, timezone.utc)
    else:
        created_dt = datetime.now(timezone.utc)

    return created_dt.isoformat()


def calculate_age_minutes(token):
    """Calculate age in minutes"""
    created_at = token.get("created_timestamp", 0)
    if created_at:
        if created_at > 1000000000000:
            created_dt = datetime.fromtimestamp(created_at / 1000, timezone.utc)
        else:
            created_dt = datetime.fromtimestamp(created_at, timezone.utc)

        age_minutes = (datetime.now(timezone.utc) - created_dt).total_seconds() / 60
    else:
        age_minutes = 0

    return round(age_minutes, 1)


if __name__ == "__main__":
    asyncio.run(get_fresh_active_tokens())
