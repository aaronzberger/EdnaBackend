'''
Generate the area.json file containing all nodes and ways
'''
import json
import os

import overpass
from termcolor import colored

from src.config import BASE_DIR

print(colored('Please wait. This query takes ~ 2m 30s for Squirrel Hill...', color='yellow'))
print('Querying Overpass API...', end=' ')

api = overpass.API(endpoint='https://overpass.kumi.systems/api/interpreter')

# Fetch all ways and nodes in Squirrel Hill
response = api.get(
    '''
    [out:json][timeout:600];
    area[name='Squirrel Hill'];
    way(area)
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
        ['access' != 'private']
        ['access' != 'no'];
    node(w);
    foreach
    {
        (
            ._;
            way(bn)
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
                ['access' != 'private']
                ['access' != 'no'];
        );
        out;
    }
    ''',
    build=False,
)
print('Response Received.\nWriting file...')
json.dump(response, open(os.path.join(BASE_DIR, 'input', 'squirrel_hill.json'), 'w', encoding='utf-8'), indent=4)
