"""
Associate houses with blocks. Take in block_output.json and generate blocks.json and houses.json
"""

# TODO: See Covode Pl and Wendover Pl (both have same-named streets, to which all the houses on pl have been wrongly assigned)

import csv
import itertools
from itertools import chain
import json
from copy import deepcopy
from dataclasses import dataclass
import os
from typing import Any, Optional
import math
from rapidfuzz import process, fuzz
import io


from src.config import (
    BASE_DIR,
    HouseInfo,
    Point,
    Block,
    street_suffixes_file,
    address_pts_file,
    block_output_file,
    blocks_file,
    blocks_file_t,
    houses_file,
    houses_file_t,
    node_coords_file,
    node_list_t,
    ALD_BUFFER,
)
from src.gps_utils import (
    along_track_distance,
    cross_track_distance,
    great_circle_distance,
)
from tqdm import tqdm

from src.viz_utils import display_blocks

MAX_DISTANCE = 500  # meters from house to segment
CHUNK_SIZE = 500  # in meters
DEBUG = True


# Create a buffer using StringIO
buffer = io.StringIO()

if DEBUG:
    with open("debug.txt", "w") as f:
        f.write("DEBUG START")

"-----------------------------------------------------------------------------------------"
"                                     Load files                                          "
"-----------------------------------------------------------------------------------------"
# region Load files
# Load the table containing node coordinates by ID
print("Loading node coordinates table...")
node_coords: dict[str, Point] = json.load(open(node_coords_file))

# Load the file (unorganized) containing house coordinates (and info)
print("Loading coordinates of houses...")
house_points_file = open(address_pts_file)
num_houses = -1
for line in house_points_file:
    num_houses += 1
house_points_file.seek(0)
house_points_reader = csv.DictReader(house_points_file)

# Load the block_output file, containing the blocks returned from the OSM query
print("Loading node and way coordinations query...")
blocks: dict[str, Any] = json.load(open(block_output_file))
# endregion

print("Loading list of street suffixes")
street_suffixes: dict[str, str] = json.load(open(street_suffixes_file))

# Map segment IDs to a dict containting the addresses and node IDs
segments_by_id: blocks_file_t = {}
houses_to_id: houses_file_t = {}


@dataclass
class Segment:
    """Define an alternate Block with more geographic information"""

    sub_node_1: Point
    sub_node_2: Point
    ctd: float  # cross track distance (to road)
    ald_offset: float  # along track distance offset (to segment bounds)
    side: bool
    id: str | None


def distance_along_path(path: node_list_t) -> float:
    """
    Find the distance through a list of Points

    Parameters:
        path (node_list_t): the navigation path to follow

    Returns:
        float: the distance through the path
    """
    distance = 0
    for first, second in itertools.pairwise(path):
        distance += great_circle_distance(
            Point(lat=first["lat"], lon=first["lon"]),  # type: ignore
            Point(lat=second["lat"], lon=second["lon"]),
        )  # type: ignore
    return distance


# We will need to temporarily store the street names for each block as well
block_to_street_names = {}

min_lat, min_lon, max_lat, max_lon = (
    40.5147085,
    -80.2215597,
    40.6199697,
    -80.0632736,
)

origin = {"lat": min_lat, "lon": min_lon}

# Calculate the number of chunks in each direction
lat_distance = great_circle_distance(origin, {"lat": max_lat, "lon": min_lon})
lon_distance = great_circle_distance(origin, {"lat": min_lat, "lon": max_lon})
num_lat_chunks = int(math.ceil(lat_distance / CHUNK_SIZE))
num_lon_chunks = int(math.ceil(lon_distance / CHUNK_SIZE))

# Create the matrix with empty lists
block_matrix: list[list[list[str]]] = [
    [list() for _ in range(num_lon_chunks)] for _ in range(num_lat_chunks)
]


# Get matrix indices for a given node
def get_matrix_index(node: Point, origin: Point, chunk_size: float) -> tuple[int, int]:
    lat_distance = great_circle_distance(
        {"lat": origin["lat"], "lon": node["lon"]}, node
    )
    lon_distance = great_circle_distance(
        {"lat": node["lat"], "lon": origin["lon"]}, node
    )
    return int(lat_distance // chunk_size), int(lon_distance // chunk_size)


print("Created empty matrix, starting to place blocks into matrix")

with tqdm(
    total=len(blocks),
    desc="Creating block location matrix",
    unit="rows",
    colour="green",
) as progress:
    for start_node in blocks:
        for block in blocks[start_node]:
            # if the block is empty, this means that the "other" start node likely contains the actual data
            if block[2] is None:
                continue
            segment_id = str(start_node) + ":" + str(block[0]) + ":" + str(block[1])

            # Create the list of sub-segments in this block
            all_node_ids = [str(i) for i in block[2]["nodes"]]
            all_nodes: list[Point] = []
            for id in all_node_ids:
                try:
                    coords = node_coords[id]
                except KeyError:
                    print(f"KeyError on finding coordinates of node {id}")
                    continue
                all_nodes.append(coords)
            for node in all_nodes:
                i, j = get_matrix_index(node, origin, CHUNK_SIZE)
                if 0 <= i < num_lat_chunks and 0 <= j < num_lon_chunks:
                    if segment_id not in block_matrix[i][j]:
                        block_matrix[i][j].append(segment_id)
        progress.update(1)


data_as_list = [
    [list(inner_set) for inner_set in inner_list] for inner_list in block_matrix
]

with open("block_matrix.json", "w") as outfile:
    json.dump(data_as_list, outfile)


# First, load every block, find subsegments, and save all data besides actual addresses to segments_by_id
with tqdm(
    total=len(blocks), desc="Reading blocks", unit="rows", colour="green"
) as progress:
    for start_node in blocks:
        for block in blocks[start_node]:
            if block[2] is None:
                continue
            segment_id = str(start_node) + ":" + str(block[0]) + ":" + str(block[1])

            # Create the list of sub-segments in this block
            all_node_ids = [str(i) for i in block[2]["nodes"]]
            all_nodes: list[Point] = []
            for id in all_node_ids:
                try:
                    coords = node_coords[id]
                except KeyError:
                    continue
                all_nodes.append(coords)

            # NOTE: For now, assume there's one way per segment
            segments_by_id[segment_id] = Block(
                addresses={},
                nodes=all_nodes,
                type=block[2]["ways"][0][1]["highway"],
            )
        progress.update(1)


def sanitize_street_name(street_name: str):
    # Split the street name by spaces
    words = street_name.casefold().split()

    if len(words) > 1:
        last_word = words[-1]

        # Check if the last word is in the lookup dictionary
        if last_word in street_suffixes:
            # If it is, replace it
            words[-1] = street_suffixes[last_word]

    # Join the words back together and return
    return " ".join(words).rstrip()


with tqdm(
    total=len(blocks), desc="Sanitizing street names", unit="rows", colour="green"
) as progress:
    for start_node in blocks:
        for block in blocks[start_node]:
            if block[2] is None:
                continue
            segment_id = str(start_node) + ":" + str(block[0]) + ":" + str(block[1])

            # Add the street name to the block
            block_to_street_names[segment_id] = list(
                set(sanitize_street_name(i[1]["name"]) for i in block[2]["ways"])
            )
        progress.update(1)

with open("block_street_mapping.json", "w") as outfile:
    json.dump(block_to_street_names, outfile)

num_failed_houses = 0


def keys_with_value(d: dict[str, list[str]], target: str) -> list[str]:
    return [key for key, values in d.items() if target in values]


def address_match_score(s1: str, s2: str, threshold=90, score_cutoff=0.0):
    # TODO: add heuristic to match w/o last word being truncated
    """
    Computes a custom score based on the Jaro distance between words in the two strings.

    Args:
    - s1, s2: The two strings to compare.
    - threshold: The minimum allowed Jaro score for two words to be considered a match.
    - score_cutoff: The minimum overall score for the function to return a non-zero result.

    Returns:
    - A score representing the ratio of matched words to the total number of words.
      Returns 0.0 immediately if the computed score is below score_cutoff.
    """
    whole_str_ratio = fuzz.ratio(s1, s2)
    if whole_str_ratio > threshold and whole_str_ratio > score_cutoff:
        return whole_str_ratio

    s1_words = s1.split()
    s2_words = s2.split()

    if len(s1_words) > 1:
        s1_words.pop()

    if len(s2_words) > 1:
        s2_words.pop()

    matched_words = 0

    for word1 in s1_words:
        for word2 in s2_words:
            # Compute the Jaro distance between word1 and word2
            jaro_score = fuzz.ratio(word1, word2)
            if jaro_score >= threshold:
                matched_words += 1

    total_words = max(len(s1_words), len(s2_words))

    score = (matched_words / total_words) * 100

    return score if score >= score_cutoff else 0.0


def search_for_best_subsegment(segment, segment_id, best_segment, house_pt: Point):
    # Iterate through each block sub-segment (block may curve)
    for node_1_data, node_2_data in itertools.pairwise(segment["nodes"]):
        node_1 = Point(**node_1_data)
        node_2 = Point(**node_2_data)

        ctd = cross_track_distance(house_pt, node_1, node_2)
        alds = along_track_distance(house_pt, node_1, node_2)

        # The distance from the point this house projects onto the line created by the segment to the segment (0 if within bounds)
        house_offset = max(
            0,
            max(alds) - great_circle_distance(node_1, node_2) - ALD_BUFFER,
        )

        # If this segment is better than the best segment, insert it
        if (
            best_segment is None
            or house_offset < best_segment.ald_offset
            or (house_offset == best_segment.ald_offset and abs(ctd) < best_segment.ctd)
        ):
            if DEBUG:
                print(
                    "Found better segment with CTD: {:.2f}".format(
                        -1 if best_segment is None else ctd
                    ),
                    file=buffer,
                )

                if best_segment is None:
                    print("Old best segment was None", file=buffer)
                else:
                    print(f"Old CTD: {best_segment.ctd}", file=buffer)
                    if house_offset < best_segment.ald_offset:
                        print(
                            f"New house_offset: {house_offset}, old: {best_segment.ald_offset}",
                            file=buffer,
                        )

            best_segment = Segment(
                sub_node_1=deepcopy(node_1),
                sub_node_2=deepcopy(node_2),
                ctd=abs(ctd),
                ald_offset=house_offset,
                side=True if ctd > 0 else False,
                id=segment_id,
            )

    return best_segment


def filter_segments(house_point):
    index = get_matrix_index(house_point, origin, CHUNK_SIZE)
    max_rows = len(block_matrix)
    max_cols = len(block_matrix[0])

    potential_indices = [
        (index[0] - 1, index[1] + 1),
        (index[0], index[1] + 1),
        (index[0] + 1, index[1] + 1),
        (index[0] - 1, index[1]),
        (index[0], index[1]),
        (index[0] + 1, index[1]),
        (index[0] - 1, index[1] - 1),
        (index[0], index[1] - 1),
        (index[0] + 1, index[1] - 1),
    ]

    valid_indices = [
        (row, col)
        for (row, col) in potential_indices
        if 0 <= row < max_rows and 0 <= col < max_cols
    ]
    filtered_ids = list(
        itertools.chain(*[block_matrix[i[0]][i[1]] for i in valid_indices])
    )
    return filtered_ids


# Next, add the houses to the blocks
with tqdm(
    total=num_houses, desc="Associating houses", unit="rows", colour="green"
) as progress:
    for item in house_points_reader:
        progress.update(1)
        # if item['municipality'].strip().upper() != 'PITTSBURGH' or \
        #         int(item['zip_code']) != 15217:
        #     continue

        # If this house is not in the area of interest, skip it
        if (
            float(item["latitude"]) < min_lat
            or float(item["latitude"]) > max_lat
            or float(item["longitude"]) < min_lon
            or float(item["longitude"]) > max_lon
        ):
            continue

        house_pt = Point(lat=float(item["latitude"]), lon=float(item["longitude"]), type="house")  # type: ignore

        address_parts = (
            item["st_premodifier"],
            item["st_prefix"],
            item["st_pretype"],
            item["st_name"],
            item["st_type"],
            item["st_postmodifier"],
        )

        raw_street_name = " ".join(part for part in address_parts if part)

        sanitized_street_name = sanitize_street_name(raw_street_name)

        best_segment: Optional[
            Segment
        ] = None  # Store the running closest segment to the house

        # First try for an exact match
        filtered_segment_ids = filter_segments(house_pt)

        if DEBUG:
            print("\n\n\n ------------------------------------------", file=buffer)
            print(
                f"Begin match for house at {house_pt['lat']}, {house_pt['lon']}",
                file=buffer,
            )

            print(
                f"Address: {item['full_address']}",
                file=buffer,
            )

            print("Filtered segment IDs in adjacent matrix locations", file=buffer)
            print(filtered_segment_ids, file=buffer)

            print(
                f"Street name to match: {sanitized_street_name} (raw: {raw_street_name})",
                file=buffer,
            )

        for segment_id in filtered_segment_ids:
            if sanitized_street_name in block_to_street_names[segment_id]:
                if DEBUG:
                    print(f"Found exact match for street name", file=buffer)
                closest_block = segments_by_id[segment_id]
                best_segment = search_for_best_subsegment(
                    closest_block, segment_id, best_segment, house_pt
                )

        filtered_block_to_street_names = {
            k: block_to_street_names[k] for k in filtered_segment_ids
        }

        # if precisely matched segment doesn't meet criteria

        if best_segment is None or best_segment.ctd > MAX_DISTANCE:
            block_names_set = set(
                chain.from_iterable(filtered_block_to_street_names.values())
            )
            if DEBUG:
                print(
                    "Failed to find exact match. Performing fuzzy search of nearby blocks",
                    file=buffer,
                )
                print(
                    f"Choices from fuzzy find for street name {sanitized_street_name}:",
                    file=buffer,
                )

                print("Set of block names to match against:", file=buffer)
                print(block_names_set, file=buffer)

            for choice in process.extract_iter(
                query=sanitized_street_name,
                choices=block_names_set,
                scorer=address_match_score,
                score_cutoff=45,
            ):
                if DEBUG:
                    print("Choice ---------", file=buffer)
                    print(choice, file=buffer)
                    print("Details of blocks with matching names:", file=buffer)
                for block_id in keys_with_value(
                    filtered_block_to_street_names, choice[0]
                ):
                    if DEBUG:
                        print("---------block---------", file=buffer)
                        print(segments_by_id[block_id], file=buffer)
                    best_segment = search_for_best_subsegment(
                        segments_by_id[block_id], block_id, best_segment, house_pt
                    )

        if best_segment is not None and best_segment.ctd <= MAX_DISTANCE:
            # Create house to insert into table
            all_points = segments_by_id[best_segment.id]["nodes"]

            sub_nodes = [
                all_points.index(best_segment.sub_node_1),
                all_points.index(best_segment.sub_node_2),
            ]

            # Calculate distances to the start and end of the block
            distance_to_start: float = 0
            distance_to_end: float = 0

            # Calculate the distance from the start of the block to the beginning of this house's sub-segment
            distance_to_start += distance_along_path(all_points[: min(sub_nodes) + 1])

            # Split this sub-segment's length between the two distances, based on this house's location
            sub_start = all_points[min(sub_nodes)]
            sub_end = all_points[max(sub_nodes)]
            distances = along_track_distance(
                p1=house_pt,
                p2=Point(lat=sub_start["lat"], lon=sub_start["lon"]),  # type: ignore
                p3=Point(lat=sub_end["lat"], lon=sub_end["lon"]),
            )  # type: ignore
            distance_to_start += distances[0]
            distance_to_end += distances[1]

            # Lastly, calculate the distance from the end of this house's sub-segment to the end of the block
            distance_to_end += distance_along_path(all_points[min(sub_nodes) + 1 :])

            output_house = HouseInfo(
                lat=house_pt["lat"],
                lon=house_pt["lon"],
                distance_to_start=round(distance_to_start),
                distance_to_end=round(distance_to_end),
                side=best_segment.side,
                distance_to_road=round(best_segment.ctd),
                subsegment=(min(sub_nodes), max(sub_nodes)),
            )
            # endregion

            # Add the house to the segments output
            segments_by_id[best_segment.id]["addresses"][
                item["full_address"]
            ] = output_house

            # Add this association to the houses file
            houses_to_id[item["full_address"]] = best_segment.id
        else:
            num_failed_houses += 1
            if DEBUG:
                print(f"Failed to associate house with point: {house_pt}", file=buffer)
                print(f'Raw street name {item["st_name"]}', file=buffer)
                print(f"Street name: {sanitized_street_name}", file=buffer)
                if best_segment is not None:
                    print(
                        f"matched against: {block_to_street_names[best_segment.id]}",
                        file=buffer,
                    )
                else:
                    print(f"street name did not find any matches", file=buffer)
                print(f"Best segment: {best_segment}", file=buffer)
                if best_segment is not None:
                    print(f"Best segment CTD: {best_segment.ctd}", file=buffer)
                print(f"MAX_DISTANCE: {MAX_DISTANCE}", file=buffer)
        if DEBUG:
            with open("debug.txt", "a") as f:
                f.write(buffer.getvalue())
            buffer.seek(0)  # set position to the start
            buffer.truncate()  # truncate the buffer from the current position

print(
    "Failed to associate {} out of {} total houses".format(
        num_failed_houses, num_houses
    )
)

# Write the output to files
print("Writing...")
json.dump(segments_by_id, open(blocks_file, "w"), indent=4)
json.dump(houses_to_id, open(houses_file, "w"), indent=4)

display_blocks(segments_by_id).save(os.path.join(BASE_DIR, "viz", "segments.html"))
