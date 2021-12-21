import overpass
import json

api = overpass.API(endpoint="https://overpass.kumi.systems/api/interpreter")

# fetch all ways and nodes
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
jsonResponse = json.dumps(response)
with open("squirrelhill.json", "w") as output_file:
    output_file.write(jsonResponse)