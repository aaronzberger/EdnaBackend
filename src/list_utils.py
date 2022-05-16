# Useful functions for walk lists
import sys
from utils import Colors, BASE_DIR, distance
import json
import os
import pickle

METERS_TO_MILES = 0.000621371192

def calculate_distance(walk_list):
    # Load the hash table containing node coordinates hashed by ID
    node_coords_table = pickle.load(open(os.path.join(BASE_DIR, 'input/hash_nodes.pkl'), 'rb'))

    data = [node_coords_table.get(int(i)) for i in walk_list['route']]
    coordinates = [(i['lat'], i['lon']) for i in data]

    running_distance = sum(
        [distance(*p1, *p2) for p1, p2 in zip(coordinates[:-1], coordinates[1:])])
    
    running_distance *= METERS_TO_MILES

    return running_distance
    

def create_walking_map(walk_list):
    def format_address(s):
        return s.lower().capitalize().replace(' ', '+')
    origin = format_address(walk_list['addresses'][0])
    destination = format_address(walk_list['addresses'][-1])
    waypoints = [format_address(i) for i in walk_list['addresses'][1:-1]]
    return 'https://www.google.com/maps/dir/?api=1&origin={}&destination={}&travelmode=walking&waypoints={}'.format(
        origin, destination, '%7C'.join(['{}'] * len(waypoints)).format(*waypoints)
    )

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(Colors.WARNING.value + 'Usage: list_utils.py [filename]' + Colors.ENDC.value)
        sys.exit()

    # Load the block_output file, containing the blocks returned from the OSM query
    print('Loading node and way coordinations query...')
    walk_list = json.load(open(os.path.join(BASE_DIR, sys.argv[1]), 'r'))

    print('''
This list is {:.2f} miles long.
See the route here: {}
    '''.format(calculate_distance(walk_list), create_walking_map(walk_list)))