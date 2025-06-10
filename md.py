import base58
import solders.pubkey as pk
import requests

# Test mint address
mint = "DgxVfMD92iBXPYpA7rJPMGFEmQKVEBXQjJX27v12pump"
prog = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"  # Pump.fun

# Generate PDA with correct seed format
BONDING_SEED = b"bonding-curve"  # Changed from "bonding"
seed = [BONDING_SEED, bytes(pk.Pubkey.from_string(mint))]
pda, _ = pk.Pubkey.find_program_address(seed, pk.Pubkey.from_string(prog))
print(f"Mint: {mint}")
print(f"Curve PDA: {pda}")

# Check if PDA exists
response = requests.post(
    "https://api.mainnet-beta.solana.com",
    json={
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getAccountInfo",
        "params": [str(pda), {"encoding": "base64"}]
    }
).json()

exists = response.get("result", {}).get("value") is not None
print(f"PDA exists: {exists}")
