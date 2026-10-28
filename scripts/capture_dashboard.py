"""Capture a screenshot of the live AgentBridge dashboard for the README."""
import sys
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8000/"
OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/dashboard.png"

with sync_playwright() as p:
    browser = p.chromium.launch()
    # Tight frame: content ends ~700px; avoid dead space below.
    page = browser.new_page(viewport={"width": 1280, "height": 720}, device_scale_factor=2)
    page.goto(URL, wait_until="networkidle")
    # let the metrics poll at least once so numbers are populated
    page.wait_for_timeout(3000)
    page.screenshot(path=OUT)
    browser.close()

print(f"Saved {OUT}")
