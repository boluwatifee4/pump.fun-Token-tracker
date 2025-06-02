#!/usr/bin/env python3
"""
🔬 Pump.fun Data Source Tester
Test each API individually to see what's working
"""

import asyncio
import aiohttp
import json

# Test token that we know exists
TEST_MINT = "2DyxvdkEM1c3D9FZk3nUHLd1NkrisiLbo1dFQC62pump"
SOLANA_RPC_URL = "https://api.mainnet-beta.solana.com"
COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd"
PUMP_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"


async def test_all_sources():
    async with aiohttp.ClientSession() as session:
        print(f"🧪 Testing data sources for token: {TEST_MINT[:8]}...\n")

        # 1. Test SOL Price
        print("1️⃣ Testing CoinGecko SOL Price...")
        sol_price = await test_sol_price(session)
        print(f"   Result: ${sol_price}\n")

        # 2. Test Token Existence
        print("2️⃣ Testing Token Existence...")
        exists = await test_token_exists(session)
        print(f"   Result: {exists}\n")

        # 3. Test Jupiter Price
        print("3️⃣ Testing Jupiter Price API...")
        jupiter_price = await test_jupiter_price(session)
        print(f"   Result: {jupiter_price}\n")

        # 4. Test DexScreener Price
        print("4️⃣ Testing DexScreener Price API...")
        dex_price = await test_dexscreener_price(session)
        print(f"   Result: {dex_price}\n")

        # 5. Test Birdeye Price
        print("5️⃣ Testing Birdeye Price API...")
        birdeye_price = await test_birdeye_price(session)
        print(f"   Result: {birdeye_price}\n")

        # 6. Test Transaction Signatures
        print("6️⃣ Testing Transaction Signatures...")
        signatures = await test_get_signatures(session)
        print(f"   Found {len(signatures)} signatures")
        if signatures:
            print(f"   Latest: {signatures[0][:16]}...")
        print()

        # 7. Test Single Transaction Details
        if signatures:
            print("7️⃣ Testing Transaction Details...")
            tx_details = await test_transaction_details(session, signatures[0])
            print(f"   Result: {tx_details}\n")

        # 8. Test Pump.fun Program Signatures
        print("8️⃣ Testing Pump.fun Program Signatures...")
        pump_sigs = await test_pump_signatures(session)
        print(f"   Found {len(pump_sigs)} pump.fun signatures")
        if pump_sigs:
            print(f"   Latest: {pump_sigs[0][:16]}...")
        print()


async def test_sol_price(session):
    try:
        async with session.get(COINGECKO_URL) as response:
            if response.status == 200:
                data = await response.json()
                return data["solana"]["usd"]
            else:
                return f"HTTP {response.status}"
    except Exception as e:
        return f"Error: {e}"


async def test_token_exists(session):
    try:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getAccountInfo",
            "params": [TEST_MINT, {"encoding": "base64"}]
        }

        async with session.post(SOLANA_RPC_URL, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                account_info = data.get("result", {}).get("value")
                return account_info is not None
            else:
                return f"HTTP {response.status}"
    except Exception as e:
        return f"Error: {e}"


async def test_jupiter_price(session):
    try:
        url = f"https://price.jup.ag/v4/price?ids={TEST_MINT}"
        async with session.get(url, timeout=10) as response:
            status = response.status
            text = await response.text()

            if status == 200:
                try:
                    data = json.loads(text)
                    token_data = data.get("data", {}).get(TEST_MINT)
                    if token_data:
                        return f"${float(token_data.get('price', 0))}"
                    else:
                        return "No price data in response"
                except:
                    return f"JSON parse error: {text[:100]}"
            else:
                return f"HTTP {status}: {text[:100]}"
    except Exception as e:
        return f"Error: {e}"


async def test_dexscreener_price(session):
    try:
        url = f"https://api.dexscreener.com/latest/dex/tokens/{TEST_MINT}"
        async with session.get(url, timeout=10) as response:
            status = response.status
            text = await response.text()

            if status == 200:
                try:
                    data = json.loads(text)
                    pairs = data.get("pairs", [])
                    if pairs:
                        price = pairs[0].get("priceUsd", 0)
                        return f"${float(price)}"
                    else:
                        return "No pairs found"
                except:
                    return f"JSON parse error: {text[:100]}"
            else:
                return f"HTTP {status}: {text[:100]}"
    except Exception as e:
        return f"Error: {e}"


async def test_birdeye_price(session):
    try:
        url = f"https://public-api.birdeye.so/public/price?address={TEST_MINT}"
        async with session.get(url, timeout=10) as response:
            status = response.status
            text = await response.text()

            if status == 200:
                try:
                    data = json.loads(text)
                    price = data.get("data", {}).get("value", 0)
                    return f"${float(price)}"
                except:
                    return f"JSON parse error: {text[:100]}"
            else:
                return f"HTTP {status}: {text[:100]}"
    except Exception as e:
        return f"Error: {e}"


async def test_get_signatures(session):
    try:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getSignaturesForAddress",
            "params": [TEST_MINT, {"limit": 10}]
        }

        async with session.post(SOLANA_RPC_URL, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                signatures = data.get("result", [])
                return [sig["signature"] for sig in signatures]
            else:
                return []
    except Exception as e:
        print(f"   Error getting signatures: {e}")
        return []


async def test_pump_signatures(session):
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
            else:
                return []
    except Exception as e:
        print(f"   Error getting pump signatures: {e}")
        return []


async def test_transaction_details(session, signature):
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
                    return {
                        "success": meta.get("err") is None,
                        "fee": meta.get("fee", 0),
                        "pre_balances": len(meta.get("preBalances", [])),
                        "post_balances": len(meta.get("postBalances", []))
                    }
                else:
                    return "No transaction data"
            else:
                return f"HTTP {response.status}"
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    asyncio.run(test_all_sources())
