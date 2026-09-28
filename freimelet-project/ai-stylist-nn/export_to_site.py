# -*- coding: utf-8 -*-
"""
After retraining (python3 train.py), run this to drop the new weights
straight into ../freimelet/index.html, replacing the old <script
type="application/json" id="aiModelData"> block. No other part of the
page is touched.

    python3 export_to_site.py [path-to-index.html]
"""
import json
import re
import sys

TARGET = sys.argv[1] if len(sys.argv) > 1 else "../freimelet/index.html"

with open("web_model.json", "r", encoding="utf-8") as f:
    model = json.load(f)
payload = json.dumps(model, ensure_ascii=False, separators=(",", ":"))

with open(TARGET, "r", encoding="utf-8") as f:
    html = f.read()

pattern = re.compile(
    r'<script type="application/json" id="aiModelData">.*?</script>',
    re.DOTALL,
)
new_tag = f'<script type="application/json" id="aiModelData">{payload}</script>'
html, n = pattern.subn(new_tag, html, count=1)

if n == 0:
    raise SystemExit("Could not find the aiModelData script tag in " + TARGET)

with open(TARGET, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Updated {TARGET} with the freshly trained model ({len(payload)} bytes).")
