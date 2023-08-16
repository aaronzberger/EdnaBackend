'''
Generate the {area}.json file containing all nodes and ways
'''
import json
import os

import overpass
from termcolor import colored

from src.config import BASE_DIR, AREA_ID, overpass_file

OVERPASS_AREA = 'Squirrel Hill'
AREA_KEY = 'squirrel_hill'

OVERPASS_AREAS = 'Bell Acres, Edgeworth'

# assert AREA_KEY == AREA_ID, 'AREA_KEY from get_data.py and AREA_ID from config.py must match'

if not os.path.exists(os.path.join(BASE_DIR, 'regions', AREA_KEY, 'input')):
    print(f'No region found called {AREA_KEY}. Creating the directory...')
    os.makedirs(os.path.join(BASE_DIR, 'regions', AREA_KEY, 'input'))

print(colored('Please wait. This query takes ~ 2m 30s for Squirrel Hill...', color='yellow'))
print('Querying Overpass API...', end=' ')

api = overpass.API(endpoint='https://overpass-api.de/api/interpreter')

# Fetch all ways and nodes in designated area
response = api.get(
    f'''
        [out:json][timeout:600];
        way(40.5147085, -80.2215597, 40.6199697, -80.0632736)
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
        way(bn)(40.5147085, -80.2215597, 40.6199697, -80.0632736)
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
