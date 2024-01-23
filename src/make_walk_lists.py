"""Run the optimizer and post-process the solution to generate walk lists."""
# TODO: Fix arbitrary large distances throughout package

from __future__ import annotations

import argparse
import sys

from termcolor import colored

from src.config import (
    CAMPAIGN_SUBSET_DB_IDX,
    NODE_COORDS_DB_IDX,
    ABODE_DB_IDX,
    PROBLEM_TYPE,
    VOTER_DB_IDX,
    DEPOT,
    NUM_ROUTES,
    TIMEOUT,
    NodeType,
    Abode,
    InternalPoint,
    Problem_Types,
    Voter,
)
from src.optimize.group_canvas import GroupCanvas
from src.optimize.turf_split import TurfSplit
from src.post_processing.post_process import (
    process_solution,
    process_partitioned_solution,
)
from src.utils.db import Database

parser = argparse.ArgumentParser(
    prog="make_walk_lists.py",
    description="Generate walk lists",
)
parser.add_argument(
    "-c",
    "--campaign_id",
    required=True,
    help="The campain id for which to select the universe",
)
args = parser.parse_args()

db = Database()

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
            colored("Abode with id {} does not have voters".format(voter["abode_id"]), color="red")
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

"-------------------------------------------------------------------------------------------------------"
"                                               Building                                                "
" Build the depots and houses for the problem and areas specified                                       "
"-------------------------------------------------------------------------------------------------------"

match PROBLEM_TYPE:
    case Problem_Types.turf_split:
        optimizer = TurfSplit(
            block_ids=block_ids, abode_ids=abode_ids, num_routes=NUM_ROUTES
        )

    case Problem_Types.group_canvas:
        try:
            result = db.get_dict(DEPOT, NODE_COORDS_DB_IDX)
        except KeyError:
            raise ValueError(
                "Depot point provided must be the id of point in the database's nodes"
            )

        depot = InternalPoint(
            lat=result["lat"],
            lon=result["lon"],
            type=NodeType.other,
            id="depot",
        )

        optimizer = GroupCanvas(
            block_ids=block_ids, abode_ids=abode_ids, depot=depot, num_routes=NUM_ROUTES
        )

    case _:
        print(colored("Invalid problem type", color="red"))
        sys.exit(1)

"-----------------------------------------------------------------------------------------"
"                                      Optimize                                           "
" Run the optimizer on the subset of the universe                                         "
"-----------------------------------------------------------------------------------------"

routes: list[list[list[InternalPoint]]] | list[list[InternalPoint]] = optimizer(
    debug=True, time_limit_s=TIMEOUT.seconds
)

"-----------------------------------------------------------------------------------------"
"                                      Post-Process                                       "
" Eliminate inaccuracies in the solution by applying some walkability and visibility      "
" heuristics (combining duplicate blocks, eliminating backtracking, etc.)                 "
" Also, generate the walk list files and visualizations                                   "
"-----------------------------------------------------------------------------------------"

if routes is None:
    print(colored("Failed to generate lists", color="red"))
    sys.exit()

print(f"Generated {len(routes)} routes")
print(routes)

if PROBLEM_TYPE == Problem_Types.turf_split:
    process_partitioned_solution(
        route_parts=routes,
        mix_distances=optimizer.mix_distances,
    )
else:
    process_solution(
        routes=routes,
        mix_distances=optimizer.mix_distances,
    )
