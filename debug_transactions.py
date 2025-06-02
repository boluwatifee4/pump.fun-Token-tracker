#!/usr/bin/env python3
"""
🔬 Transaction Debug Test
Check if we're finding ANY transactions for this specific token
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timedelta, timezone

# Test with the specific token from your CSV
TEST_MINT = "CCpTDK9dDdmvcQtpPrv6zDBWXgjqq5LrJHidJW7Jpump"
SOLANA_RPC_URL = "https://api.mainnet-beta.solana.com"
PUMP_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"


async def debug_transactions():
    async with aiohttp.ClientSession() as session:
        print(f"🔍 Debugging transactions for: {TEST_MINT[:8]}...\n")

        # 1. Check recent signatures for token
        print("1️⃣ Getting token signatures...")
        token_sigs = await get_token_signatures(session)
        print(f"   Found: {len(token_sigs)} signatures")

        if token_sigs:
            print("   Recent signatures:")
            for i, sig in enumerate(token_sigs[:5]):
                print(f"   {i+1}. {sig[:16]}...")
        print()

        # 2. Check recent pump.fun program signatures
        print("2️⃣ Getting pump.fun program signatures...")
        pump_sigs = await get_pump_signatures(session)
        print(f"   Found: {len(pump_sigs)} signatures")
        print()

        # 3. Analyze first transaction in detail
        if token_sigs:
            print("3️⃣ Analyzing first transaction...")
            tx_analysis = await analyze_transaction(session, token_sigs[0])
            print(f"   Result: {json.dumps(tx_analysis, indent=2)}")
            print()

        # 4. Check for pump.fun interactions
        if token_sigs:
            print("4️⃣ Checking for pump.fun interactions...")
            pump_interactions = await check_pump_interactions(session, token_sigs[:10])
            print(f"   Pump.fun interactions found: {pump_interactions}")
            print()

        # 5. Check timestamps of recent activity
        print("5️⃣ Checking transaction timestamps...")
        await check_transaction_times(session, token_sigs[:10])


async def get_token_signatures(session):
    try:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getSignaturesForAddress",
            "params": [TEST_MINT, {"limit": 50}]
        }

        async with session.post(SOLANA_RPC_URL, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                signatures = data.get("result", [])
                return [sig["signature"] for sig in signatures]
    except Exception as e:
        print(f"   Error: {e}")
    return []


async def get_pump_signatures(session):
    try:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getSignaturesForAddress",
            "params": [PUMP_PROGRAM, {"limit": 20}]
        }

        async with session.post(SOLANA_RPC_URL, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                signatures = data.get("result", [])
                return [sig["signature"] for sig in signatures]
    except Exception as e:
        print(f"   Error: {e}")
    return []


async def analyze_transaction(session, signature):
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

        async with session.post(SOLANA_RPC_URL, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                tx = data.get("result")

                if tx:
                    meta = tx.get("meta", {})
                    transaction = tx.get("transaction", {})
                    message = transaction.get("message", {})

                    # Check for pump.fun program
                    account_keys = message.get("accountKeys", [])
                    pump_program_found = any(
                        acc.get("pubkey") == PUMP_PROGRAM for acc in account_keys
                    )

                    # Get balance changes
                    pre_balances = meta.get("preBalances", [])
                    post_balances = meta.get("postBalances", [])

                    sol_changes = []
                    if len(pre_balances) == len(post_balances):
                        for i in range(len(pre_balances)):
                            change = (
                                post_balances[i] - pre_balances[i]) / 1_000_000_000
                            sol_changes.append(round(change, 6))

                    return {
                        "signature": signature[:16] + "...",
                        "success": meta.get("err") is None,
                        "fee": meta.get("fee", 0),
                        "pump_program_involved": pump_program_found,
                        "accounts_count": len(account_keys),
                        # First 5 accounts
                        "sol_balance_changes": sol_changes[:5],
                        "block_time": tx.get("blockTime"),
                        "recent": tx.get("blockTime", 0) > (datetime.now().timestamp() - 3600) if tx.get("blockTime") else False
                    }

    except Exception as e:
        return {"error": str(e)}

    return {"error": "No transaction data"}


async def check_pump_interactions(session, signatures):
    pump_count = 0
    for sig in signatures:
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTransaction",
                "params": [sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}]
            }

            async with session.post(SOLANA_RPC_URL, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    tx = data.get("result")

                    if tx:
                        message = tx.get("transaction", {}).get("message", {})
                        account_keys = message.get("accountKeys", [])

                        pump_found = any(
                            acc.get("pubkey") == PUMP_PROGRAM for acc in account_keys
                        )

                        if pump_found:
                            pump_count += 1

        except:
            continue

    return pump_count


async def check_transaction_times(session, signatures):
    times = []
    now = datetime.now().timestamp()

    for sig in signatures[:5]:
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getTransaction",
                "params": [sig, {"encoding": "base64"}]
            }

            async with session.post(SOLANA_RPC_URL, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    tx = data.get("result")

                    if tx and tx.get("blockTime"):
                        block_time = tx["blockTime"]
                        age_hours = (now - block_time) / 3600
                        times.append({
                            "signature": sig[:16] + "...",
                            "block_time": datetime.fromtimestamp(block_time).strftime("%Y-%m-%d %H:%M:%S"),
                            "age_hours": round(age_hours, 1)
                        })

        except:
            continue

    if times:
        print("   Recent transaction times:")
        for t in times:
            print(
                f"   {t['signature']} | {t['block_time']} | {t['age_hours']}h ago")
    else:
        print("   No transaction times found")

if __name__ == "__main__":
    asyncio.run(debug_transactions())
