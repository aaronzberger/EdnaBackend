'''
Generate the area.json file containing all nodes and ways
'''
import json
import overpass
import os

from utils import BASE_DIR, Colors


OUPTUT_NAME = 'squirrel_hill'

print(Colors.WARNING.value + 'Please wait. This query takes 2m 30s for Squirrel Hill...' + Colors.ENDC.value)
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
jsonResponse = json.dumps(response)

# Create the input directory if it doesn't exist
os.makedirs(os.path.join(BASE_DIR, 'input'), exist_ok=True)

with open(os.path.join(BASE_DIR, 'input/{}.json').format(OUPTUT_NAME), 'w') as output_file:
    output_file.write(jsonResponse)
