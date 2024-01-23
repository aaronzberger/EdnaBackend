"""
Pre-process the data necessary for a campaign: populate the distance matrices with all the necessary data.

TODO: In all distance matrices, add storage matrix to only search for blocks, nodes, houses that actually might be within
the maximum storage distance. Will speed up to linear time.
TODO Add general write buffer class to block and node distances to speed up writing to the database (as is in house distances hard-coded)
For this, also write to buffer after certain size, instead of writing at the end.
"""

# TODO: Implement some checks on the universal data to make sure it's valid for this campaign??

from __future__ import annotations

import argparse
import sys

from termcolor import colored

from src.config import (
    CAMPAIGN_SUBSET_DB_IDX,
    ABODE_DB_IDX,
    VOTER_DB_IDX,
    Abode,
    Voter,
)
from src.distances.blocks import BlockDistances
from src.distances.nodes import NodeDistances
from src.utils.db import Database

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

# Walk up the database from voters to abodes to blocks
# To retrieve all nodes, take the first and last node from all blocks with abodes which have voters in this campaign
voter_ids = db.get_set(args.campaign_id, CAMPAIGN_SUBSET_DB_IDX)

print(f"Found {len(voter_ids)} voters")

abode_ids: set[str] = set()
for voter in voter_ids:
    voter: Voter = db.get_dict(voter, VOTER_DB_IDX)
    if voter is None:
        print(colored("Voter {} not found in database".format(voter), color="red"))
        sys.exit(1)

    # Ensure the corresponding abode has this voter
    abode: Abode = db.get_dict(voter["abode_id"], ABODE_DB_IDX)

    if "voter_ids" not in abode:
        print(
            colored("Abode {} does not have voters".format(voter["abode_id"]), color="red")
        )
        sys.exit(1)

    abode_ids.add(voter["abode_id"])

print(f"Found {len(abode_ids)} abode")

block_ids: set[str] = set()
for voter in abode_ids:
    abode_data = db.get_dict(voter, ABODE_DB_IDX)
    if abode_data is None:
        print(colored("Abode {} not found in database".format(voter), color="red"))
        sys.exit(1)
    block_ids.add(abode_data["block_id"])

print(f"Found {len(block_ids)} blocks")

# Populate the node distance matrix
node_distances = NodeDistances(block_ids=block_ids, skip_update=True)

# Populate the block distance matrix
block_distances = BlockDistances(block_ids=block_ids, node_distances=node_distances, skip_update=True)
