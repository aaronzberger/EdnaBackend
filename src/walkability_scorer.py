import itertools
import json
from typing import Any

from src.config import blocks_file, blocks_file_t
from src.gps_utils import Point
from src.route import get_distance
from src.timeline_utils import SubSegment


_blocks: blocks_file_t = json.load(open(blocks_file))


def score(start: Point, end: Point, input: list[SubSegment]) -> dict[str, Any]:
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
    for first, second in itertools.pairwise([start] + list(itertools.chain.from_iterable([s.houses for s in input])) + [end]):
        total += get_distance(first, second)

    # Calculate crossings
    for segment in input:
        segment_type = _blocks[segment.segment.id]['type']
        for first, second in itertools.pairwise(segment.houses):
            if _blocks[segment.segment.id]['addresses'][first.id]['side'] != _blocks[segment.segment.id]['addresses'][second.id]['side']:
                try:
                    road_crossings[segment_type] += 1
                except KeyError:
                    road_crossings['other'] += 1

    # Calculate number of segments
    unique_segment_ids: set[str] = set()
    for seg in input:
        if seg.segment.id not in unique_segment_ids:
            unique_segment_ids.add(seg.segment.id)

    return {
        'road_crossings': road_crossings,
        'segments': len(unique_segment_ids),
        'max_dist': max(distances_from_depot),
        'distance': total,
        'num_houses': len(list(itertools.chain.from_iterable([i.houses for i in input])))
    }
