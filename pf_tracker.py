import asyncio
import aiohttp
import csv
import os
import json
import base64
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from solana.rpc.async_api import AsyncClient
from solders.pubkey import Pubkey
from solders.signature import Signature

MORALIS_KEY = os.getenv("MORALIS_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJub25jZSI6ImE0ZjhjYWE0LTdjNDctNDgzZS1iMjQ4LTNjNmIwMWQ4NGI5YiIsIm9yZ0lkIjoiNDUwNDcwIiwidXNlcklkIjoiNDYzNDkyIiwidHlwZUlkIjoiNzI5MjdmMjAtMDQ3YS00YTJmLWE1NzEtMTZiOTlkNjlmNWFiIiwidHlwZSI6IlBST0pFQ1QiLCJpYXQiOjE3NDg4MTk2MzEsImV4cCI6NDkwNDU3OTYzMX0.pm2_wlQdxBLPyp_W6MfGjyUkwH4bw2ochrqARAPgcu0")

# CSV where we dump every 10 s sample:
CSV_PATH = "all_token_tracks.csv"

# Keep track of active tasks & their start times
active_samplers = {}      # mint → asyncio.Task
start_times = {}          # mint → datetime of launch

# Add Pump.fun program constants
PUMPFUN_PROGRAM_ID = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
PUMPFUN_GLOBAL = "4wTV1YmiEkRvAtNtsSGPtUrqRYQMe5SKy2uB4Gp1wnCx"
PUMPFUN_FEE = "CebN5WGQ4jvEPvsVU4EoHEpgzq1VV3kGhhZp4zFBo8Jp"

# -----------------------------------------------------------------------------


async def fetch_new_pump_tokens(session):
    """Call Moralis's 'Get New Pump.fun Tokens' endpoint."""
    url = f"https://solana-gateway.moralis.io/token/mainnet/exchange/pumpfun/new?limit=100"
    try:
        async with session.get(url, headers={"X-API-Key": MORALIS_KEY}) as resp:
            if resp.status != 200:
                print(f"API Error: {resp.status} - {await resp.text()}")
                return []
            data = await resp.json()
            print(f"Raw API response: {data}")  # Debug line
            
            # Moralis returns {"result": [...]} format
            if isinstance(data, dict) and 'result' in data:
                return data['result']
            else:
                print(f"Unexpected response format: {type(data)}")
                return []
    except Exception as e:
        print(f"Error fetching new tokens: {e}")
        return []


async def fetch_bonding_status(session, mint):
    """Check a token's bondingProgress via Moralis."""
    url = f"https://solana-gateway.moralis.io/token/mainnet/{mint}/bonding-status"
    try:
        async with session.get(url, headers={"X-API-Key": MORALIS_KEY}) as resp:
            if resp.status == 200:
                data = await resp.json()
                return float(data.get("bondingProgress", 0.0))
            return 0.0
    except Exception as e:
        print(f"Error fetching bonding status for {mint}: {e}")
        return 0.0


async def fetch_graduated_tokens(session):
    """Call Moralis's 'Get Graduated Pump.fun Tokens' endpoint."""
    url = f"https://solana-gateway.moralis.io/token/mainnet/exchange/pumpfun/graduated?limit=100"
    try:
        async with session.get(url, headers={"X-API-Key": MORALIS_KEY}) as resp:
            if resp.status != 200:
                print(f"Graduated API Error: {resp.status} - {await resp.text()}")
                return []
            data = await resp.json()
            print(f"Graduated tokens response: {data}")  # Debug line
            
            # Moralis returns {"result": [...]} format
            if isinstance(data, dict) and 'result' in data:
                return data['result']
            else:
                return []
    except Exception as e:
        print(f"Error fetching graduated tokens: {e}")
        return []


async def fetch_bonding_tokens(session):
    """Call Moralis's 'Get Bonding Pump.fun Tokens' endpoint."""
    url = f"https://solana-gateway.moralis.io/token/mainnet/exchange/pumpfun/bonding?limit=100"
    try:
        async with session.get(url, headers={"X-API-Key": MORALIS_KEY}) as resp:
            if resp.status != 200:
                print(f"Bonding API Error: {resp.status} - {await resp.text()}")
                return []
            data = await resp.json()
            print(f"Bonding tokens response: {data}")  # Debug line
            
            # Moralis returns {"result": [...]} format
            if isinstance(data, dict) and 'result' in data:
                return data['result']
            else:
                return []
    except Exception as e:
        print(f"Error fetching bonding tokens: {e}")
        return []


# Add RPC client
RPC_URL = "https://api.mainnet-beta.solana.com"


async def get_signatures_for_mint(mint, since_datetime):
    """Get recent transaction signatures for a token mint"""
    try:
        client = AsyncClient(RPC_URL)
        
        # Convert datetime to unix timestamp
        since_timestamp = int(since_datetime.timestamp())
        
        # Convert string address to Pubkey object
        mint_pubkey = Pubkey.from_string(mint)
        
        # Get signatures for the token account
        response = await client.get_signatures_for_address(
            mint_pubkey,  # ← Use Pubkey object instead of string
            limit=100,
        )
        
        if response.value:
            # Filter by timestamp
            recent_sigs = []
            for sig_info in response.value:
                if sig_info.block_time and sig_info.block_time >= since_timestamp:
                    recent_sigs.append(str(sig_info.signature))  # Convert signature to string
            
            await client.close()
            return recent_sigs
        
        await client.close()
        return []
        
    except Exception as e:
        print(f"Error getting signatures for {mint}: {e}")
        return []


async def get_transaction_details(signature):
    """Get detailed transaction data for analysis"""
    try:
        client = AsyncClient(RPC_URL)
        
        # Convert signature string to Signature object if needed
        sig_obj = Signature.from_string(signature)
        
        response = await client.get_transaction(
            sig_obj,  # ← Use Signature object
            encoding="json",
            max_supported_transaction_version=0
        )
        
        await client.close()
        
        if response.value:
            return response.value
        return None
        
    except Exception as e:
        print(f"Error getting transaction {signature}: {e}")
        return None
    


async def parse_pump_transaction(tx_data, mint):
    """Parse a Pump.fun transaction to extract buy/sell data"""
    try:
        if not tx_data or not tx_data.get('transaction'):
            return None
            
        meta = tx_data.get('meta', {})
        if meta.get('err'):
            return None  # Skip failed transactions
            
        transaction = tx_data['transaction']
        message = transaction.get('message', {})
        instructions = message.get('instructions', [])
        
        # Look for Pump.fun program interactions
        for instruction in instructions:
            program_id_index = instruction.get('programIdIndex')
            if program_id_index is not None:
                accounts = message.get('accountKeys', [])
                if program_id_index < len(accounts):
                    program_id = accounts[program_id_index]
                    if program_id == PUMPFUN_PROGRAM_ID:
                        return await extract_pump_trade_data(tx_data, mint)
        
        return None
        
    except Exception as e:
        print(f"Error parsing transaction: {e}")
        return None


async def extract_pump_trade_data(tx_data, mint):
    """Extract trade data from Pump.fun transaction"""
    try:
        meta = tx_data.get('meta', {})
        pre_balances = meta.get('preBalances', [])
        post_balances = meta.get('postBalances', [])
        pre_token_balances = meta.get('preTokenBalances', [])
        post_token_balances = meta.get('postTokenBalances', [])
        
        # Find SOL balance changes
        sol_change = 0
        if len(pre_balances) > 0 and len(post_balances) > 0:
            # Typically account[0] is the signer
            sol_change = (post_balances[0] - pre_balances[0]) / 1e9  # Convert lamports to SOL
        
        # Find token balance changes
        token_change = 0
        for pre_bal in pre_token_balances:
            if pre_bal.get('mint') == mint:
                for post_bal in post_token_balances:
                    if (post_bal.get('mint') == mint and 
                        post_bal.get('accountIndex') == pre_bal.get('accountIndex')):
                        pre_amount = float(pre_bal.get('uiTokenAmount', {}).get('uiAmount', 0))
                        post_amount = float(post_bal.get('uiTokenAmount', {}).get('uiAmount', 0))
                        token_change = post_amount - pre_amount
                        break
        
        # Determine if buy or sell
        is_buy = token_change > 0 and sol_change < 0
        is_sell = token_change < 0 and sol_change > 0
        
        if not (is_buy or is_sell):
            return None
            
        # Get signer (wallet address)
        message = tx_data['transaction']['message']
        accounts = message.get('accountKeys', [])
        signer = accounts[0] if accounts else None
        
        # Get timestamp
        block_time = tx_data.get('blockTime', 0)
        
        return {
            'signer': signer,
            'is_buy': is_buy,
            'is_sell': is_sell,
            'sol_amount': abs(sol_change),
            'token_amount': abs(token_change),
            'timestamp': block_time,
            'signature': tx_data.get('transaction', {}).get('signatures', [None])[0]
        }
        
    except Exception as e:
        print(f"Error extracting trade data: {e}")
        return None


async def get_sol_price_usd():
    """Get current SOL price from CoinGecko"""
    try:
        async with aiohttp.ClientSession() as session:
            url = "https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd"
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return float(data.get('solana', {}).get('usd', 0))
        return 0
    except:
        return 0


# Replace the placeholder collect_metrics function:
async def collect_metrics(mint, since, last_R, solana_session):
    """Collect actual 13 metrics from Solana blockchain"""
    
    # Get token account transactions since last interval
    signatures = await get_signatures_for_mint(mint, since)
    
    # Initialize metrics
    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "R": 0.0,
        "A": 0.0,
        "unique_buyers": 0,
        "top3_pct": 0.0,
        "lp_burn": False,
        "price_sol": 0.0,
        "price_usd": 0.0,
        "mcap_usd": 0.0,
        "whale_flag": False,
        "buy_sell_delta": 0.0,
        "buy_pressure": 0.0,
        "sell_pressure": 0.0,
        "bot_like": False
    }
    
    # Get bonding progress first (this always works)
    try:
        bonding_progress = await fetch_bonding_status(solana_session, mint)
        metrics["R"] = bonding_progress / 100.0
    except:
        pass
    
    if not signatures:
        # Calculate A (acceleration) even without new transactions
        metrics["A"] = metrics["R"] - last_R
        return metrics
    
    # Parse transactions for actual data
    buyers = set()
    sellers = set()
    whale_wallets = set()
    total_buys = 0.0
    total_sells = 0.0
    sol_flows = []
    trades = []
    buyer_volumes = defaultdict(float)
    
    current_time = datetime.now(timezone.utc).timestamp()
    
    for sig in signatures[:50]:  # Limit to avoid rate limits
        try:
            tx_data = await get_transaction_details(sig)
            if tx_data:
                trade_data = await parse_pump_transaction(tx_data, mint)
                if trade_data:
                    trades.append(trade_data)
                    signer = trade_data['signer']
                    sol_amount = trade_data['sol_amount']
                    timestamp = trade_data['timestamp']
                    
                    if trade_data['is_buy']:
                        buyers.add(signer)
                        total_buys += sol_amount
                        buyer_volumes[signer] += sol_amount
                        sol_flows.append(sol_amount)
                        
                        # Whale detection (>= 1 SOL in last 20 seconds)
                        if sol_amount >= 1.0 and (current_time - timestamp) <= 20:
                            whale_wallets.add(signer)
                            
                    elif trade_data['is_sell']:
                        sellers.add(signer)
                        total_sells += sol_amount
                        sol_flows.append(-sol_amount)
        except Exception as e:
            print(f"Error processing transaction {sig}: {e}")
            continue
    
    # Calculate metrics
    metrics["unique_buyers"] = len(buyers)
    metrics["buy_pressure"] = total_buys
    metrics["sell_pressure"] = total_sells
    metrics["buy_sell_delta"] = total_buys - total_sells
    metrics["whale_flag"] = len(whale_wallets) > 0
    
    # Top 3 buyer concentration
    if buyer_volumes:
        top3_volumes = sorted(buyer_volumes.values(), reverse=True)[:3]
        total_buy_volume = sum(buyer_volumes.values())
        if total_buy_volume > 0:
            metrics["top3_pct"] = sum(top3_volumes) / total_buy_volume * 100
    
    # Bot detection (>= 4 buys of < 0.1 SOL in 0.2s span)
    small_buys = [t for t in trades if t['is_buy'] and t['sol_amount'] < 0.1]
    if len(small_buys) >= 4:
        timestamps = [t['timestamp'] for t in small_buys]
        timestamps.sort()
        for i in range(len(timestamps) - 3):
            if timestamps[i+3] - timestamps[i] <= 0.2:
                metrics["bot_like"] = True
                break
    
    # Calculate A (acceleration) - change in net flow
    current_flow = sum(sol_flows)
    metrics["A"] = metrics["R"] - last_R  # Change in bonding progress
    
    # Price calculation (simplified - use bonding curve formula)
    if metrics["R"] > 0:
        # Pump.fun bonding curve: rough approximation
        # Real formula would need exact curve parameters
        progress_sol = metrics["R"] * 85  # 85 SOL at 100%
        if progress_sol > 0:
            # Simplified price calculation
            metrics["price_sol"] = progress_sol / 1000000  # Rough estimate
            
            # Get SOL price for USD conversion
            sol_price_usd = await get_sol_price_usd()
            metrics["price_usd"] = metrics["price_sol"] * sol_price_usd
            
            # Market cap (assuming 1B token supply)
            token_supply = 1000000000
            metrics["mcap_usd"] = metrics["price_usd"] * token_supply
    
    return metrics


async def sampler_task(mint, created_at, session):
    """
    Every 10 s for up to 45 min (or until bondingProgress ≥ 100), collect metrics.
    """
    start = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    end = start + timedelta(minutes=45)
    interval_start = start
    last_R = 0.0

    # Ensure CSV has header:
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "mint", "progress", "R", "A", "unique_buyers", "top3_pct",
                "lp_burn", "price_sol", "price_usd", "mcap_usd",
                "whale_flag", "buy_sell_delta", "buy_pressure", "sell_pressure", "bot_like"
            ])
            writer.writeheader()

    while datetime.now(timezone.utc) < end:
        # 1) Check bondingProgress
        prog = await fetch_bonding_status(session, mint)
        # 2) Collect your 13 metrics via whatever existing code you have:
        #    (e.g. trades from RPC, DexScreener price, Coalgebra logic, etc.)
        row = await collect_metrics(
            mint=mint,
            since=interval_start,
            last_R=last_R,
            solana_session=session  # if you need RPC calls via aiohttp, etc.
        )
        row["mint"] = mint
        row["progress"] = round(prog, 2)

        # Append to CSV:
        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)

        last_R = row["R"]
        interval_start = datetime.now(timezone.utc)

        # Stop early if progress ≥ 100:
        if prog >= 100:
            print(
                f"[{datetime.now(timezone.utc)}] 🎉 {mint} finished bonding (100 %).")
            break

        await asyncio.sleep(10)

    # Mark sampler as done:
    active_samplers.pop(mint, None)
    print(f"[{datetime.now(timezone.utc)}] ✅ sampler for {mint} ended.")


# -----------------------------------------------------------------------------
async def migration_watcher(session):
    """Watch for tokens migrating to pools"""
    seen = set()
    while True:
        try:
            # Check graduated tokens
            grads = await fetch_graduated_tokens(session)
            for obj in grads:
                mint = obj.get("tokenAddress")
                graduated_at = obj.get("graduatedAt")
                
                if mint and mint not in seen:
                    # This token has migrated!
                    print(f"🚀 MIGRATION DETECTED: {mint} at {graduated_at}")
                    seen.add(mint)
                    
                    # Cancel tracking if still running
                    if mint in active_samplers:
                        task = active_samplers[mint]
                        task.cancel()
                        print(f"✅ Stopped tracking {mint} - migration complete")
                        
        except Exception as e:
            print(f"Error in migration watcher: {e}")
            
        await asyncio.sleep(30)


# -----------------------------------------------------------------------------
async def main():
    async with aiohttp.ClientSession() as session:
        # 1) At startup, load any existing bonding tokens
        try:
            bonding_list = await fetch_bonding_tokens(session)
            
            for obj in bonding_list:
                if not isinstance(obj, dict):
                    print(f"Skipping invalid token object: {obj}")
                    continue
                    
                # Use correct field names from Moralis API
                mint = obj.get("tokenAddress")
                created_at = obj.get("createdAt")
                
                if not mint or not created_at:
                    print(f"Missing required fields in token: {obj}")
                    continue
                    
                if mint in active_samplers:
                    continue

                # Calculate remaining time
                created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                end = created + timedelta(minutes=45)
                rem = (end - datetime.now(timezone.utc)).total_seconds()
                if rem <= 0:
                    continue

                # Spawn sampler
                task = asyncio.create_task(sampler_task(mint, created_at, session))
                active_samplers[mint] = task
                start_times[mint] = created
                print(f"Started tracking existing bonding token: {mint}")
                
        except Exception as e:
            print(f"Error loading existing bonding tokens: {e}")

        # 2) Start migration watcher
        asyncio.create_task(migration_watcher(session))
        
        # 3) Main polling loop
        while True:
            try:
                new_list = await fetch_new_pump_tokens(session)
                
                for obj in new_list:
                    if not isinstance(obj, dict):
                        print(f"Skipping invalid new token object: {obj}")
                        continue
                        
                    # Use correct field names from Moralis API
                    mint = obj.get("tokenAddress")
                    created_at = obj.get("createdAt")
                    
                    if not mint or not created_at:
                        print(f"Missing required fields in new token: {obj}")
                        continue
                        
                    if mint in active_samplers:
                        continue

                    # Calculate remaining time
                    created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    rem = (created + timedelta(minutes=45) - datetime.now(timezone.utc)).total_seconds()
                    if rem <= 0:
                        continue

                    print(f"[{datetime.now(timezone.utc)}] ➕ Detected new Pump.fun token {mint} launched at {created_at}")
                    
                    # Start tracking
                    t = asyncio.create_task(sampler_task(mint, created_at, session))
                    active_samplers[mint] = t
                    start_times[mint] = created

            except Exception as e:
                print(f"Error in main polling loop: {e}")

            await asyncio.sleep(15)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped by user")
