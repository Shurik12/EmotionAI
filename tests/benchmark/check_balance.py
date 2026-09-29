#!/usr/bin/env python3
"""Check Multitool API key balance and spend."""
import json
import os
import sys
import urllib.request

API_KEY = os.environ.get("MULTITOOL_API_KEY")
if not API_KEY:
    print("ERROR: MULTITOOL_API_KEY not set")
    sys.exit(1)

req = urllib.request.Request(
    "https://shared1.multitool.works:4000/user/info",
    headers={"Authorization": f"Bearer {API_KEY}"},
)
resp = urllib.request.urlopen(req)
data = json.loads(resp.read())

keys = sorted(data.get("keys", []), key=lambda x: x.get("spend", 0), reverse=True)

print("=" * 72)
print(f"  MULTITOOL API — User: {data.get('user_id', 'N/A')}")
print(f"  Total Keys: {len(keys)}")
print("=" * 72)
print()
print(f"  {'Alias':<20s} {'Spent':>10s} {'Budget':>10s} {'Used':>6s} {'Status':>8s}  Last Active")
print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*6} {'-'*8}  {'-'*19}")
for k in keys[:15]:
    alias = (k.get("key_alias") or "?")[:18]
    spend = k.get("spend", 0)
    budget = k.get("max_budget", 0)
    pct = (spend / budget * 100) if budget else 0
    status = "BLOCKED" if k.get("blocked") else "OK"
    last = (k.get("last_active") or "")[:19]
    print(f"  {alias:<20s} ${spend:>8.2f} ${budget:>8.0f} {pct:>5.1f}% {status:>8}  {last}")
if len(keys) > 15:
    print(f"  ... and {len(keys) - 15} more keys")
print()

# Find and highlight the current key (matching by alias or partial token)
for k in keys:
    token = k.get("token", "")
    key_name = k.get("key_name", "")
    if "...g8ng" in key_name or (token and API_KEY in token):
        print(f"  ╔═══ YOUR KEY ═══════════════════════════════════════════╗")
        print(f"  ║  {k.get('key_alias', '?'):<47s} ║")
        print(f"  ║  Spend:    ${k['spend']:>8.2f} / ${k['max_budget']:>8.0f} ({k['spend']/k['max_budget']*100:>5.1f}%)  ║")
        print(f"  ║  Status:   {'Active' if not k.get('blocked') else 'BLOCKED':<40s} ║")
        print(f"  ║  Created:  {k.get('created_at', '?')[:19]:<40s} ║")
        print(f"  ║  Active:   {k.get('last_active', '?')[:19]:<40s} ║")
        print(f"  ╚══════════════════════════════════════════════════════╝")
        break
