"""Probe the Gemini free-tier rate limit with rapid sequential calls."""
import json
import os
import re
import time
import urllib.error
import urllib.request

key = os.environ["GEMINI_API_KEY"]
model = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

ok = fail = 0
for i in range(7):
    body = json.dumps(
        {"contents": [{"role": "user", "parts": [{"text": f"ping {i}"}]}]}
    ).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()
        ok += 1
        print(f"  call {i + 1}: OK   {time.time() - t0:.1f}s")
    except urllib.error.HTTPError as e:
        fail += 1
        msg = e.read().decode()
        tier = "FREE TIER" if "free_tier" in msg else ("PAID" if "paid" in msg else "?")
        print(f"  call {i + 1}: {e.code} ({tier}) {time.time() - t0:.1f}s")
        m = re.search(r'"retryDelay":\s*"([\d.]+)', msg)
        if m:
            print(f"     server says wait {m.group(1)}s")

print(f"--- {ok} ok / {fail} failed out of 7 rapid calls")
