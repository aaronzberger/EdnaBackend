'''
Save a json that maps node IDs to their GPS coordinates
'''

import json

from tqdm import tqdm

from src.config import overpass_file, NODE_COORDS_DB_IDX, STYLE_COLOR
from src.utils.db import Database


# Returned from OSM query of all nodes and ways in a region
loaded = json.load(open(overpass_file))

Database()

for item in tqdm(loaded['elements'], desc='Writing node coordinates', colour=STYLE_COLOR, unit='nodes'):
    if item['type'] == 'node':
        # NOTE: This is the only ID casting from int to str. Downstream, all IDs are ints
        item_id = str(item['id'])

        if not Database.exists(item_id, NODE_COORDS_DB_IDX):
            Database.set_dict(item_id, {
                'lat': item['lat'],
                'lon': item['lon']
            }, NODE_COORDS_DB_IDX)
