"""
Save a json that maps node IDs to their GPS coordinates
"""

import json

from tqdm import tqdm

from src.config import WriteablePoint, overpass_file, NODE_COORDS_DB_IDX, STYLE_COLOR
from src.utils.db import Database


# Returned from OSM query of all nodes and ways in a region
loaded = json.load(open(overpass_file))

db = Database()

for item in tqdm(
    loaded["elements"],
    desc="Writing node coordinates",
    colour=STYLE_COLOR,
    unit="nodes",
):
    if item["type"] == "node":
        # NOTE: This is the only ID casting from int to str. Downstream, all IDs are strings
        item_id = str(item["id"])

        point = WriteablePoint(
            lat=item["lat"],
            lon=item["lon"],
        )

        db.set_dict(item_id, dict(point), NODE_COORDS_DB_IDX)
