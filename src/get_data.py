'''Generates the json file containing all nodes and ways'''

import overpass
import json

api = overpass.API(endpoint="https://overpass.kumi.systems/api/interpreter")

# Fetch all ways and nodes in Squirrel Hill
print('Querying Overpass API...', end=' ')
response = api.get(
    """
    [out:json][timeout:600];
    area[name="Squirrel Hill"];
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
    """,
    build=False,
)
print('Response Received.\nWriting file...')
jsonResponse = json.dumps(response)
with open("test.json", "w") as output_file:
    output_file.write(jsonResponse)