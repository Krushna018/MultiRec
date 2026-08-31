
"""Optional helper to download MovieLens 1M for a real-data extension.

MovieLens 1M contains 1,000,209 ratings. This script is optional because the core
project is fully runnable offline using the included deterministic synthetic generator.
"""
from pathlib import Path
import urllib.request, zipfile, io

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

print("Downloading MovieLens 1M...")
raw = urllib.request.urlopen(URL).read()
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    z.extractall(DATA)
print("Extracted to", DATA / "ml-1m")
