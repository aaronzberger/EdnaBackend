import json
from sys import argv
import pickle
import os
import numpy as np

from utils import BASE_DIR, lat_lon_to_x_y, x_y_to_lat_lon

if len(argv) != 3:
    print('Usage: convert_list_to_qgis.py [WALK LIST FILE] [DESTINATION FILE]')

walk_list = json.load(open(argv[1], 'r'))

output = {}
output['type'] = 'FeatureCollection'
output['name'] = os.path.splitext(os.path.basename(argv[2]))[0]
output['crs'] = {'type': 'name', 'properties': {'name': 'urn:ogc:def:crs:OGC:1.3:CRS84'}}
output['features'] = []

# Load the hash table containing node coordinates hashed by ID
print('Loading hash table of nodes...')
node_coords_table = pickle.load(open(os.path.join(BASE_DIR, 'input/hash_nodes.pkl'), 'rb'))

# First, add each of the houses to the visualization
for house in walk_list['addresses']:
    properties = {'Address': house[0]}
    geometry = {'type': 'Point', 'coordinates': [float(house[2]), float(house[1])]}
    item = {'type': 'Feature', 'properties': properties, 'geometry': geometry}
    output['features'].append(item)

# Next, add each segment to the visualization
coordinates = []
for node in walk_list['route']:
    house = node_coords_table.get(int(node))
    coordinates.append([house["lon"], house["lat"]])

geometry = {'type': 'LineString', 'coordinates': coordinates}
properties = {}
item = {'type': 'Feature', 'properties': properties, 'geometry': geometry}
output['features'].append(item)

# Get the first two GPS coordinates of the route
first_coords = node_coords_table.get(int(walk_list['route'][0]))
second_coords = node_coords_table.get(int(walk_list['route'][1]))

# Convert to XY
x1, y1, zone_num, zone_let = lat_lon_to_x_y(first_coords['lat'], first_coords['lon'])
x2, y2, zone_num1, zone_let1 = lat_lon_to_x_y(second_coords['lat'], second_coords['lon'])

# Calculate back polygon line
start_vec = np.array([x1, y1])
end_vec = np.array([x2, y2])
delta = np.subtract(end_vec, start_vec)
delta /= np.linalg.norm(delta)
mid_point = np.subtract(start_vec, delta * 10)

slope = (y2 - y1) / (x2 - x1)
perpendicular_slope = np.array([1, -1 / slope])
perpendicular_slope /= np.linalg.norm(perpendicular_slope)

p1 = x_y_to_lat_lon(*np.add(mid_point, perpendicular_slope * 10).tolist(), zone_num, zone_let)
p2 = x_y_to_lat_lon(*np.subtract(mid_point, perpendicular_slope * 10).tolist(), zone_num1, zone_let1)
coordinates = [[first_coords['lon'], first_coords['lat']],
               p1[::-1], p2[::-1],
               [first_coords['lon'], first_coords['lat']]]

geometry = {'type': 'LineString', 'coordinates': coordinates}
properties = {'name': 'start'}
item = {'type': 'Feature', 'properties': properties, 'geometry': geometry}
output['features'].append(item)

# Do the same for the end of the list
second_to_last_coords = node_coords_table.get(int(walk_list['route'][-2]))
last_coords = node_coords_table.get(int(walk_list['route'][-1]))

# Convert to XY
x1, y1, zone_num, zone_let = lat_lon_to_x_y(second_to_last_coords['lat'], second_to_last_coords['lon'])
x2, y2, zone_num1, zone_let1 = lat_lon_to_x_y(last_coords['lat'], last_coords['lon'])

# Calculate back polygon line
start_vec = np.array([x1, y1])
end_vec = np.array([x2, y2])
delta = np.subtract(end_vec, start_vec)
delta /= np.linalg.norm(delta)
poly_point = np.add(end_vec, delta * 10)

slope = (y2 - y1) / (x2 - x1)
perpendicular_slope = np.array([1, -1 / slope])
perpendicular_slope /= np.linalg.norm(perpendicular_slope)

p1 = x_y_to_lat_lon(*np.add(end_vec, perpendicular_slope * 10).tolist(), zone_num, zone_let)
p2 = x_y_to_lat_lon(*np.subtract(end_vec, perpendicular_slope * 10).tolist(), zone_num1, zone_let1)
poly_pt = x_y_to_lat_lon(*poly_point.tolist(), zone_num1, zone_let1)
coordinates = [[last_coords['lon'], last_coords['lat']],
               p1[::-1], poly_pt[::-1], p2[::-1],
               [last_coords['lon'], last_coords['lat']]]

geometry = {'type': 'LineString', 'coordinates': coordinates}
properties = {'name': 'end'}
item = {'type': 'Feature', 'properties': properties, 'geometry': geometry}
output['features'].append(item)

# Write the final walk list
print('Writing walk list to {}'.format(argv[2]))
with open(argv[2], 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=4)
