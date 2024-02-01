"""
Associate abodes with blocks. Take in block_output.json and write blocks and abodes to the database
"""

# TODO: See Covode Pl and Wendover Pl (both have same-named streets, to which all the abodes on pl have been wrongly assigned)

import csv
from decimal import Decimal
import itertools
from itertools import chain
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional
import math
from rapidfuzz import process, fuzz
import io
import uuid

from src.config import (
    UUID_NAMESPACE,
    AbodeGeography,
    Abode,
    NodeType,
    InternalPoint,
    Block,
    WriteablePoint,
    street_suffixes_file,
    address_pts_file,
    geocoded_universe_file,
    block_output_file,
    ALD_BUFFER,
    AREA_BBOX,
    BLOCK_DB_IDX,
    NODE_COORDS_DB_IDX,
    ABODE_DB_IDX,
)
from src.utils.gps import (
    along_track_distance,
    cross_track_distance,
    distance_along_path,
    great_circle_distance,
    pt_to_utm,
)

from src.utils.db import Database

from src.utils.address import Address

from tqdm import tqdm

from src.utils.viz import display_blocks

MAX_DISTANCE = 500  # meters from abode to segment

# In meters, the square size of each chunk in the matrix
# Higher values will result in faster processing, but more false positives in matching
CHUNK_SIZE = 500

DEBUG = False

# Create a buffer using StringIO
buffer = io.StringIO()

# Initialize the database (connect)
db = Database()
block_dict = {}

if DEBUG:
    with open("debug.txt", "w") as f:
        f.write("DEBUG START")

"-----------------------------------------------------------------------------------------"
"                                     Load files                                          "
"-----------------------------------------------------------------------------------------"
# region Load files

# Load the file (unorganized) containing abode coordinates (and info)
print("Loading coordinates of abodes...")
abode_points_file = open(geocoded_universe_file)
num_abodes = -1
for _ in abode_points_file:
    num_abodes += 1
abode_points_file.seek(0)
abode_points_reader = csv.DictReader(abode_points_file)

# Load the block_output file, containing the blocks returned from the OSM query
print("Loading node and way coordinations query...")
blocks: dict[str, Any] = json.load(open(block_output_file))
# endregion

print("Loading list of street suffixes")
street_suffixes: dict[str, str] = json.load(open(street_suffixes_file))

# NOTE/TODO: This file is temporary, and will be eliminated when we transition to abode keys
# addresses_to_id: addresses_file_t = {}

# NOTE/TODO: This file is also temporary, and is only used for manual matching (eliminated with abode keys)
# id_to_addresses: dict[str, dict[str, str]] = {}

# reverse_geocode: list[tuple[float, float, dict[str, str]]] = []


@dataclass
class Segment:
    """Define an alternate Block with more geographic information."""
    sub_node_1: InternalPoint
    sub_node_2: InternalPoint
    ctd: float  # cross track distance (to road)
    ald_offset: float  # along track distance offset (to segment bounds)
    side: bool
    id: str | None


# We will need to temporarily store the street names for each block as well
block_to_street_names = {}

min_lat, min_lon, max_lat, max_lon = AREA_BBOX

origin = InternalPoint(lat=min_lat, lon=min_lon, type=NodeType.other, id="origin")

# Calculate the number of chunks in each direction
lat_distance = great_circle_distance(
    origin, InternalPoint(lat=max_lat, lon=min_lon, type=NodeType.other, id="max_lat")
)
lon_distance = great_circle_distance(
    origin, InternalPoint(lat=min_lat, lon=max_lon, type=NodeType.other, id="max_lon")
)

num_lat_chunks = int(math.ceil(lat_distance / CHUNK_SIZE))
num_lon_chunks = int(math.ceil(lon_distance / CHUNK_SIZE))

# Create the matrix with empty lists
block_matrix: list[list[list[str]]] = [
    [list() for _ in range(num_lon_chunks)] for _ in range(num_lat_chunks)
]


# Get matrix indices for a given node
def get_matrix_index(
    node: InternalPoint, origin: InternalPoint, chunk_size: float
) -> tuple[int, int]:
    """
    Get the matrix indices for a given node.

    Parameters
    ----------
        node: The node to get the matrix indices for.
        origin: The origin node.
        chunk_size: The size of each chunk in the matrix.

    Returns
    -------
        A tuple containing the matrix indices for the given node.
    """
    lat_distance = great_circle_distance(
        InternalPoint(lat=origin["lat"], lon=node["lon"], type=NodeType.other, id=""),
        node,
    )
    lon_distance = great_circle_distance(
        InternalPoint(lat=node["lat"], lon=origin["lon"], type=NodeType.other, id=""),
        node,
    )
    return int(lat_distance // chunk_size), int(lon_distance // chunk_size)


print("Created empty matrix, starting to place blocks into matrix")
counter = 0
with tqdm(
    total=len(blocks),
    desc="Creating block location matrix",
    unit="rows",
    colour="green",
) as progress:
    for block in blocks:
        segment_id = block['id']
        for node in block['nodes']:
            i, j = get_matrix_index(node, origin, CHUNK_SIZE)
            if 0 <= i < num_lat_chunks and 0 <= j < num_lon_chunks:
                if segment_id not in block_matrix[i][j]:
                    block_matrix[i][j].append(segment_id)
        progress.update(1)

# First, load every block, find subsegments, and save all data besides actual addresses to segments_by_id
with tqdm(
    total=len(blocks), desc="Reading blocks", unit="rows", colour="green"
) as progress:
    for block in blocks:
        block_to_write = Block(
            id=block['id'],
            abodes={},
            nodes=block['nodes'],
            type=block['type'],
        )
        block_dict[block['id']] = block_to_write

        progress.update(1)

with tqdm(
    total=len(blocks), desc="Sanitizing street names", unit="rows", colour="green"
) as progress:
    for block in blocks:
        segment_id = block['id']
        # Add the street name to the block
        if block['street_name'] is not None:
            block_to_street_names[segment_id] = list(
                set(
                    [Address.sanitize_street_name(block['street_name'])]
                )
            )
        else:
            block_to_street_names[segment_id] = list(set([]))
        progress.update(1)

num_failed_abodes = 0


def keys_with_value(d: dict[str, list[str]], target: str) -> list[str]:
    return [key for key, values in d.items() if target in values]


def address_match_score(s1: str, s2: str, threshold=90, score_cutoff=0.0):
    # TODO: add heuristic to match w/o last word being truncated
    """
    Compute a custom score based on the Jaro distance between words in the two strings.

    Parameters
    ----------
        s1, s2: The two strings to compare.
        threshold: The minimum allowed Jaro score for two words to be considered a match.
        score_cutoff: The minimum overall score for the function to return a non-zero result.

    Returns
    -------
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


def search_for_best_subsegment(
    segment, segment_id, best_segment, abode_point: InternalPoint
):
    # Iterate through each block sub-segment (block may curve)
    # print(segment['nodes'])
    for node_1_data, node_2_data in itertools.pairwise(segment["nodes"]):
        node_1 = InternalPoint(**node_1_data)
        node_2 = InternalPoint(**node_2_data)

        ctd = cross_track_distance(abode_point, node_1, node_2)
        alds = along_track_distance(abode_point, node_1, node_2)

        # The distance from the point this abode projects onto the line created by the segment to the segment (0 if within bounds)
        abode_offset = max(
            0,
            max(alds) - great_circle_distance(node_1, node_2) - ALD_BUFFER,
        )

        # Convert the nodes to UTM
        x1, y1, _, _ = pt_to_utm(node_1)
        x2, y2, _, _ = pt_to_utm(node_2)
        xh, yh, _, _ = pt_to_utm(abode_point)
        cross_product = (x2 - x1) * (yh - y1) - (y2 - y1) * (xh - x1)

        abode_side = cross_product > 0

        # If this segment is better than the best segment, insert it
        if (
            best_segment is None
            or abode_offset < best_segment.ald_offset
            or (abode_offset == best_segment.ald_offset and abs(ctd) < best_segment.ctd)
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
                    if abode_offset < best_segment.ald_offset:
                        print(
                            f"New abode_offset: {abode_offset}, old: {best_segment.ald_offset}",
                            file=buffer,
                        )
            best_segment = Segment(
                sub_node_1=deepcopy(node_1),
                sub_node_2=deepcopy(node_2),
                ctd=abs(ctd),
                ald_offset=abode_offset,
                side=bool(abode_side),
                id=segment_id,
            )
    # print(best_segment)
    return best_segment


def filter_segments(abode_point):
    index = get_matrix_index(abode_point, origin, CHUNK_SIZE)
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


# Next, add the abodes to the blocks
with tqdm(
    total=num_abodes, desc="Associating abodes", unit="rows", colour="green"
) as progress:
    for item in abode_points_reader:
        progress.update(1)
        # if item['municipality'].strip().upper() != 'PITTSBURGH' or \
        #         int(item['zip_code']) != 15217:
        #     continue

        # If this abode is not in the area of interest, skip it
    
        if "latitude" in item.keys() and (
            float(item["latitude"]) < min_lat
            or float(item["latitude"]) > max_lat
            or float(item["longitude"]) < min_lon
            or float(item["longitude"]) > max_lon
        ):
            continue
        
        abode_point = InternalPoint(lat=float(item["latitude"]), lon=float(item["longitude"]), type=NodeType.abode, id="")

        street_name_parts = (
            item["Street Name"],
        )

        raw_street_name = " ".join(part for part in street_name_parts if part)

        sanitized_street_name = Address.sanitize_street_name(raw_street_name)

        # address_pts: addr_num_prefix,addr_num,addr_num_suffix,st_premodifier,st_prefix,st_pretype,st_name,st_type,st_postmodifier,
        # unit_type,unit,floor,municipality,county,state,zip_code
        # as far as I can tell, there is never any data in addr_num_prefix
        formatted_address: Address = Address(
            item["House Number"],
            item["House Number Suffix"],
            sanitized_street_name,
            item["Apartment Number"],
            None,
            None,  # TODO: add function to sanitize state names
            item["Zip"],
        )

        best_segment: Optional[
            Segment
        ] = None  # Store the running closest segment to the abode

        # First try for an exact match
        filtered_segment_ids = filter_segments(abode_point)

        if DEBUG:
            print("\n\n\n ------------------------------------------", file=buffer)
            print(
                f"Begin match for abode at {abode_point['lat']}, {abode_point['lon']}",
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
                    print("Found exact match for street name", file=buffer)
                closest_block = block_dict[segment_id]
                # closest_block = segments_by_id[segment_id]
                best_segment = search_for_best_subsegment(
                    closest_block, segment_id, best_segment, abode_point
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
                        # print(segments_by_id[block_id], file=buffer)
                    # print('calling from 485???')
                    best_segment = search_for_best_subsegment(
                        block_dict[block_id],
                        block_id,
                        best_segment,
                        abode_point,
                    )

        if best_segment is not None and best_segment.ctd <= MAX_DISTANCE:
            # Create abode to insert into table
            # print(best_segment)
            # print(block_dict[best_segment.id])
            all_points = block_dict[best_segment.id]["nodes"]
            # all_points = segments_by_id[str(best_segment.id)]["nodes"]
            sub_nodes = [
                all_points.index(best_segment.sub_node_1),
                all_points.index(best_segment.sub_node_2),
            ]

            # Calculate distances to the start and end of the block
            distance_to_start: float = 0
            distance_to_end: float = 0

            # Calculate the distance from the start of the block to the beginning of this abode's sub-segment
            distance_to_start += distance_along_path(all_points[: min(sub_nodes) + 1])

            # Split this sub-segment's length between the two distances, based on this abode's location
            sub_start = all_points[min(sub_nodes)]
            sub_end = all_points[max(sub_nodes)]
            distances = along_track_distance(
                p1=abode_point,
                p2=InternalPoint(
                    lat=sub_start["lat"],
                    lon=sub_start["lon"],
                    type=NodeType.node,
                    id="",
                ),
                p3=InternalPoint(
                    lat=sub_end["lat"], lon=sub_end["lon"], type=NodeType.node, id=""
                ),
            )
            distance_to_start += distances[0]
            distance_to_end += distances[1]

            # Lastly, calculate the distance from the end of this abode's sub-segment to the end of the block
            distance_to_end += distance_along_path(all_points[min(sub_nodes) + 1:])
            
            display_address = str(int(float(item["House Number"]))) + " " + item["Street Name"]
            # Add this association to the abodes file
            lat_rounded = Decimal(str(abode_point["lat"])).quantize(Decimal("0.0001"))
            lon_rounded = Decimal(str(abode_point["lon"])).quantize(Decimal("0.0001"))
            uuid_input = display_address + str(lat_rounded) + str(lon_rounded)

            abode_uuid = uuid.uuid5(UUID_NAMESPACE, uuid_input)

            abode_geography = AbodeGeography(
                id=str(abode_uuid),
                point=WriteablePoint(
                    lat=abode_point["lat"],
                    lon=abode_point["lon"]
                ),
                distance_to_start=round(distance_to_start),
                distance_to_end=round(distance_to_end),
                side=best_segment.side,
                distance_to_road=round(best_segment.ctd),
                subsegment_start=min(sub_nodes),
                subsegment_end=max(sub_nodes),
            )

            abode_semantics = Abode(
                id=str(abode_uuid),
                display_address=display_address,
                block_id=str(best_segment.id),
                city=item["City"],
                state=item["State"],
                zip=item["Zip"],
            )
            counter += 1
            # Add the abode to the block (Note that we expect the block to already exist in the database most of the time)
            old_block = block_dict[best_segment.id]
            old_block["abodes"][str(abode_uuid)] = abode_geography

            # Add the abode to the abodes file
            db.set_dict(str(abode_uuid), dict(abode_semantics), ABODE_DB_IDX)

        else:
            num_failed_abodes += 1
            if DEBUG:
                print(f"Failed to associate abode with point: {abode_point}", file=buffer)
                print(f'Raw street name {item["st_name"]}', file=buffer)
                print(f"Street name: {sanitized_street_name}", file=buffer)
                if best_segment is not None:
                    print(
                        f"matched against: {block_to_street_names[best_segment.id]}",
                        file=buffer,
                    )
                else:
                    print("street name did not find any matches", file=buffer)
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
    "Failed to associate {} out of {} total abodes".format(
        num_failed_abodes, num_abodes
    )
)
db.set_multiple_dict(block_dict, BLOCK_DB_IDX)
display_blocks()
