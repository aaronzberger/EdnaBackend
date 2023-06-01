import itertools
from typing import Any
import json
from decimal import Decimal
from termcolor import colored

from src.gps_utils import Point, SubBlock
from src.route import get_distance
from src.config import houses_file_t, houses_file, blocks_file, blocks_file_t, Block, HouseInfo, ROAD_WIDTH


_blocks: blocks_file_t = json.load(open(blocks_file))


def decimalize(x) -> Decimal:
    return Decimal(x).quantize(Decimal('0.0001'))


def get_house_info(block: Block, house: Point) -> HouseInfo | None:
    '''
    Get the house info for a house in a block

    Parameters:
        block (Block): the block
        house (Point): the house

    Returns:
        HouseInfo: the house info
    '''
    lat, lon = decimalize(house['lat']), decimalize(house['lon'])
    return next(filter(lambda x, lat=lat, lon=lon: decimalize(x['lat']) == lat and decimalize(x['lon']) == lon, block['addresses'].values()), None)


def score(input: list[SubBlock]) -> dict[str, Any]:
    road_crossings = {
        'motorway': 0,
        'trunk': 0,
        'primary': 0,
        'secondary': 0,
        'tertiary': 0,
        'unclassified': 0,
        'residential': 0,
        'service': 0,
        'other': 0
    }
    depot_point = input[0].start
    all_house_points = list(itertools.chain.from_iterable(lis.houses for lis in input))
    distances_from_depot = [get_distance(i, depot_point) for i in all_house_points]

    # Calculate route distance
    total = 0.0
    for first, second in itertools.pairwise(
            [input[0].start] + list(itertools.chain.from_iterable([s.houses for s in input])) + [input[-1].end]):
        total += get_distance(first, second)

    # Calculate crossings
    for block in input:
        block_type = block.block['type']
        for first, second in itertools.pairwise(block.houses):
            # Associate the house points with the houses in the block
            first_house = get_house_info(block.block, first)
            second_house = get_house_info(block.block, second)

            if first_house is None or second_house is None:
                print(colored('Error: address {} or {} not found in block {}'.format(first, second, block.block['id']), color='yellow'))
                continue

            if first_house['side'] != second_house['side']:
                try:
                    road_crossings[block_type] += 1
                except KeyError:
                    road_crossings['other'] += 1

            if first_house['side'] != second_house['side']:
                total += first_house['distance_to_road'] + second_house['distance_to_road']
            else:
                addition = first_house['distance_to_road'] + second_house['distance_to_road'] - ROAD_WIDTH[block_type]
                if addition < 10:
                    print(colored('Warning: distance between {} and {} is less than 10 meters'.format(first, second), color='yellow'))
                total += addition

    return {
        'road_crossings': road_crossings,
        'max_dist': max(distances_from_depot),
        'distance': total,
        'num_houses': len(list(itertools.chain.from_iterable([i.houses for i in input])))
    }


if __name__ == '__main__':
    import csv
    import sys

    if len(sys.argv) != 2:
        print('Usage: python3 walkability_scorer.py <input_file>')
        exit(1)

    # Read in CSV file
    with open(sys.argv[1], 'r') as f:
        reader = csv.DictReader(f)
        data = [row for row in reader]

    # Read in houses file
    with open(houses_file, 'r') as f:
        houses: houses_file_t = json.load(f)

    # Associate houses with blocks
    walk_list: list[SubBlock] = []
    current_block_id = None
    current_sub_block = None
    for row in data:
        address = row['House'] + ' ' + row['Street'].upper()
        try:
            block_id = houses[address]
        except KeyError:
            print('Could not find address: ' + address)
            continue
        if block_id != current_block_id:
            if current_sub_block is not None:
                walk_list.append(current_sub_block)
            current_block_id = block_id
            current_sub_block = SubBlock(block=_blocks[block_id], start=_blocks[block_id]['nodes'][0], end=_blocks[block_id]['nodes'][-1],
                                         extremum=None, houses=[], navigation_points=_blocks[block_id]['nodes'])

        house_address: HouseInfo | None = next(filter(lambda x, address=address: x == address, _blocks[block_id]['addresses'].keys()), None)
        house = _blocks[block_id]['addresses'][house_address]
        if house is None:
            print(colored('Error: address {} not found in block {}'.format(address, block_id), color='yellow'))
            continue
        current_sub_block.houses.append(Point(lat=house['lat'], lon=house['lon']))

    # Add last sub block
    if current_sub_block is not None:
        walk_list.append(current_sub_block)

    # Score
    scored = score(walk_list)

    print(str(scored['num_houses']) + ',' + str(scored['distance']) + ',' + str(scored['road_crossings']['tertiary']) + ',' +
          str(scored['road_crossings']['secondary']) + ',' + str(scored['road_crossings']['residential']))
