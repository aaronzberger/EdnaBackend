"""
Pre-process the data necessary for a campaign: populate the distance matrices with all the necessary data.
"""

# TODO: Implement some checks on the universal data to make sure it's valid for this campaign??

"""Run the optimizer and post-process the solution to generate walk lists."""
# TODO: Fix arbitrary large distances throughout package

from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import Optional

from sklearn.cluster import DBSCAN, AgglomerativeClustering
from termcolor import colored

from src.config import (
    BLOCK_DB_IDX,
    CAMPAIGN_SUBSET_DB_IDX,
    NODE_COORDS_DB_IDX,
    NUM_LISTS,
    PLACE_DB_IDX,
    PROBLEM_TYPE,
    SUPER_CLUSTERING,
    VOTER_DB_IDX,
    DEPOTS,
    NodeType,
    PlaceSemantics,
    Point,
    Problem_Types,
    blocks_file_t,
    clustering_pickle_file,
    optimizer_points_pickle_file,
    pt_id,
)
from src.distances.blocks import BlockDistances
from src.distances.houses import HouseDistances
from src.distances.mix import MixDistances
from src.distances.nodes import NodeDistances
from src.optimize.group_canvas import GroupCanvas
from src.optimize.optimizer import Optimizer
from src.optimize.turf_split import TurfSplit
# from src.post_processing.post_process import process_solution
from src.utils.db import Database
from src.utils.viz import display_clustered_blocks

parser = argparse.ArgumentParser(
    prog="prepare_campaign_universe.py",
    description="Prepare the data necessary for a campaign",
)
parser.add_argument(
    "-c",
    "--campaign_id",
    required=True,
    help="The campain id for which to select the universe",
)
args = parser.parse_args()

db = Database()

# TODO: Here, we'll also create the campaign subset with some input data, likely? For now, this is manual.

# Walk up the database from voters to places to blocks
# To retrieve all nodes, take the first and last node from all blocks with places which have voters in this campaign
voter_ids = db.get_set(args.campaign_id, CAMPAIGN_SUBSET_DB_IDX)

print(f"Found {len(voter_ids)} voters")

place_ids: set[str] = set()
for voter in voter_ids:
    place = db.get_dict(voter, VOTER_DB_IDX)
    if place is None:
        print(colored("Voter {} not found in database".format(voter), color="red"))
        sys.exit(1)

    # Ensure the corresponding place has this voter
    this_place: PlaceSemantics = db.get_dict(place["place"], PLACE_DB_IDX)

    if "voters" not in this_place:
        print(
            colored("Place {} does not have voters".format(place["place"]), color="red")
        )
        sys.exit(1)

    place_ids.add(place["place"])

print(f"Found {len(place_ids)} places")

block_ids: set[str] = set()
for place in place_ids:
    place_data = db.get_dict(place, PLACE_DB_IDX)
    if place_data is None:
        print(colored("Place {} not found in database".format(place), color="red"))
        sys.exit(1)
    block_ids.add(place_data["block_id"])

print(f"Found {len(block_ids)} blocks")

# Populate the node distance matrix
node_distances = NodeDistances(
    block_ids=block_ids, skip_update=False
)

# Populate the block distance matrix
block_distances = BlockDistances(
    block_ids=block_ids, node_distances=node_distances, skip_update=False
)

# Populate the house distance matrix
house_distances = HouseDistances(
    block_ids=block_ids, node_distances=node_distances, skip_update=False
)
