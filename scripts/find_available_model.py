"""Find a Gemini model that still has free-tier quota right now.

Quotas are tracked per model, so gemini-3.6-flash being exhausted does not
mean the others are. One cheap call each; first success wins.
"""
import json
import os
import re
import time
import urllib.error
import urllib.request

key = os.environ["GEMINI_API_KEY"]
CANDIDATES = [
    "gemini-3.5-flash",
    "gemini-3.7-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
    "gemini-3.6-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-3.8-flash",
]

working = []
for model in CANDIDATES:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )
    body = json.dumps(
        {"contents": [{"role": "user", "parts": [{"text": "say OK"}]}]}
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
        print(f"  {model:26s} OK    {time.time() - t0:5.1f}s   <-- HAS QUOTA")
        working.append(model)
    except urllib.error.HTTPError as e:
        msg = e.read().decode()
        if e.code == 429:
            m = re.search(r'"retryDelay":\s*"([\d.]+)', msg)
            wait = f"{float(m.group(1)) / 3600:.1f}h" if m else "?"
            print(f"  {model:26s} 429   {time.time() - t0:5.1f}s   (wait {wait})")
        elif e.code == 404:
            print(f"  {model:26s} 404   {time.time() - t0:5.1f}s   (not available)")
        else:
            print(f"  {model:26s} {e.code}  {time.time() - t0:5.1f}s")
    except Exception as e:  # noqa: BLE001
        print(f"  {model:26s} {type(e).__name__} {time.time() - t0:5.1f}s")

print()
if working:
    print(f"RECOMMEND: GEMINI_MODEL={working[0]}")
else:
    print("No model has quota right now.")
