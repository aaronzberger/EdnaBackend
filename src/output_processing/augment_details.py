# Read in the details file and the output files and add some additional information

import os
import sys
import csv
import json
from typing import Any
from src.config import InternalPoint, coords_node_file, blocks_file, block_output_file
from decimal import Decimal
from src.utils.gps import great_circle_distance


coords_node = json.load(open(coords_node_file))
blocks = json.load(open(blocks_file))
block_output = json.load(open(block_output_file))


# Read in the details file
details_file = sys.argv[1]
details: dict[str, Any] = json.load(open(details_file))

output_file = details_file.replace(".json", "_augmented.json")

output_dir = sys.argv[2]


def find_street_name(id_1, id_2):
    for block in block_output[id_1]:
        if block[0] == int(id_2) and block[2] is not None:
            return block[2]["ways"][0][1]["name"]
    for block in block_output[id_2]:
        if block[0] == int(id_1) and block[2] is not None:
            return block[2]["ways"][0][1]["name"]
    return None


for list_id in details.keys():
    # Read in the output file
    list_file = os.path.join(output_dir, f"{list_id}.json")
    output: dict[str, Any] = json.load(open(list_file))

    # Add the details
    num_voters = 0
    if "blocks" not in output:
        print(list_id, "has no blocks")
    for block in output["blocks"]:
        for house in block["abodes"]:
            num_voters += len(house["voter_info"])

    details[list_id]["num_voters"] = num_voters

    # Find the intersection name (the two streets)
    # First, find the node ids by the start point
    start_point = details[list_id]["start_point"]

    # Round lat and lon to 4 decimal abodes
    lat = Decimal(start_point["lat"]).quantize(Decimal("0.0001"))
    lon = Decimal(start_point["lon"]).quantize(Decimal("0.0001"))

    try:
        potential_ids = coords_node[f"{lat}:{lon}"]
        if len(potential_ids) == 1:
            node_id = potential_ids[0][0]
        else:
            node_id = min(potential_ids, key=lambda x: great_circle_distance(InternalPoint(lat=lat, lon=lon), InternalPoint(lat=x[1]["lat"], lon=x[1]["lon"])))[0]
    except KeyError:
        print(f"Could not find node for {lat}:{lon}")
        continue

    # print(f"Found node {node_id} for {lat}:{lon}")

    # This id will exist in exactly two block names
    block_names = []
    for block_name in blocks.keys():
        if node_id in block_name:
            block_names.append(block_name)

    street_names = []

    for block_name in block_names:
        # For the first block id, find the street name
        b1_id1, b1_id2, _ = block_name.split(':')

        street_name_1 = find_street_name(b1_id1, b1_id2)

        if street_name_1 is not None:
            street_names.append(street_name_1)

    # Remove duplicates from street names
    street_names = list(set(street_names))

    print(street_names)

    # Join with and
    if len(street_names) == 2:
        details[list_id]["street_name"] = f"{street_names[0]} and {street_names[1]}"
    else:
        if len(street_names) < 2:
            details[list_id]["street_name"] = input("Enter the street name for start point " + str(start_point) + ": ")
        else:
            preliminary = f"{street_names[0]} and {street_names[1]}"
            # Ask for user input
            approval = input(f"For street names {street_names}, using {preliminary}. Is this correct? (y/n) ")
            if approval == "y":
                details[list_id]["street_name"] = preliminary
            else:
                details[list_id]["street_name"] = input("Enter the street name: ")


# Write out the details file
json.dump(details, open(output_file, "w"), indent=4)

print("Wrote output to ", output_file)
