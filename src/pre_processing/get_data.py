'''
Generate the overpass.json file containing all nodes and ways
'''
import json
import os

import overpass
from termcolor import colored

from src.config import BASE_DIR, AREA_ID, overpass_file, AREA_BBOX

if not os.path.exists(os.path.join(BASE_DIR, 'regions', AREA_ID, 'input')):
    print(f'No region found called {AREA_ID}. Creating the directory...')
    os.makedirs(os.path.join(BASE_DIR, 'regions', AREA_ID, 'input'))

print(colored('Please wait. This query takes ~ 2m 30s for Squirrel Hill...', color='yellow'))
print('Querying Overpass API...', end=' ')

api = overpass.API(endpoint='https://overpass-api.de/api/interpreter')

# Fetch all ways and nodes in designated area

# TODO: Replace bounding box with polygon
response = api.get(
    f'''
        [out:json][timeout:600];
        way({AREA_BBOX[0]}, {AREA_BBOX[1]}, {AREA_BBOX[2]}, {AREA_BBOX[3]})
          ['name']
          ['highway']
          ['highway' != 'path']
          ['highway' != 'steps']
          ['highway' != 'motorway']
          ['highway' != 'motorway_link']
          ['highway' != 'raceway']
          ['highway' != 'bridleway']
          ['highway' != 'proposed']
          ['highway' != 'construction']
          ['highway' != 'elevator']
          ['highway' != 'bus_guideway']
          ['highway' != 'footway']
          ['highway' != 'cycleway']
          ['foot' != 'no']
          ['access' != 'no'];
        node(w);
        foreach
        {{
        (
        ._;
        way(bn)({AREA_BBOX[0]}, {AREA_BBOX[1]}, {AREA_BBOX[2]}, {AREA_BBOX[3]})
        ['name']
        ['highway']
        ['highway' != 'path']
        ['highway' != 'steps']
        ['highway' != 'motorway']
        ['highway' != 'motorway_link']
        ['highway' != 'raceway']
        ['highway' != 'bridleway']
        ['highway' != 'proposed']
        ['highway' != 'construction']
        ['highway' != 'elevator']
        ['highway' != 'bus_guideway']
        ['highway' != 'footway']
        ['highway' != 'cycleway']
        ['foot' != 'no']
        ['access' != 'no'];
        );
        out;
        }}
    ''',
    build=False,
)
print('Response Received.\nWriting file...')
json.dump(response, open(overpass_file, 'w', encoding='utf-8'), indent=4)
