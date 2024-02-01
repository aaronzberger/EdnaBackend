"""
Generate the overpass.json file containing all nodes and ways
"""
import json
import os

import overpass
from termcolor import colored

from src.config import BASE_DIR, CAMPAIGN_ID, overpass_file, AREA_BBOX

if not os.path.exists(os.path.join(BASE_DIR, "regions", CAMPAIGN_ID, "input")):
    print(f"No region found called {CAMPAIGN_ID}. Creating the directory...")
    os.makedirs(os.path.join(BASE_DIR, "regions", CAMPAIGN_ID, "input"))

print(
    colored(
        "Please wait. This query takes ~ 2m 30s for Squirrel Hill...", color="yellow"
    )
)
print("Querying Overpass API...", end=" ")

api = overpass.API(endpoint="https://overpass-api.de/api/interpreter")

# Fetch all ways and nodes in designated area

# TODO: Replace bounding box with polygon
response = api.get(
    f"""
    [out:json][timeout:600];
 way(40.5147085, -80.2215597, 40.6199697, -80.0632736)
	["name"]["highway"]; out; node (w); out body;
    """,
    build=False,
)
print("Response Received.\nWriting file...")
json.dump(response, open(overpass_file, "w", encoding="utf-8"), indent=4)
