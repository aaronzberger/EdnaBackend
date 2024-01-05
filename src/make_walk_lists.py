"""Run the optimizer and post-process the solution to generate walk lists."""
# TODO: Fix arbitrary large distances throughout package

from __future__ import annotations

import argparse
import pickle
import sys

from termcolor import colored

from src.config import (
    CAMPAIGN_SUBSET_DB_IDX,
    NODE_COORDS_DB_IDX,
    PLACE_DB_IDX,
    PROBLEM_TYPE,
    VOTER_DB_IDX,
    DEPOT,
    NodeType,
    PlaceSemantics,
    Point,
    Problem_Types,
    optimizer_points_pickle_file,
)
from src.optimize.group_canvas import GroupCanvas
from src.optimize.turf_split import TurfSplit
# from src.post_processing.post_process import process_solution
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
parser.add_argument(
    "-n",
    "--no-optimize",
    action="store_true",
    help="Skip the optimization step, post-process only",
)
parser.add_argument(
    "-r",
    "--restart",
    action="store_true",
    help="Force-perform the optimization on all clusters",
)
args = parser.parse_args()

db = Database()

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

"-------------------------------------------------------------------------------------------------------"
"                                               Building                                                "
" Build the depots and houses for the problem and areas specified                                       "
"-------------------------------------------------------------------------------------------------------"

match PROBLEM_TYPE:
    case Problem_Types.turf_split:
        optimizer = TurfSplit(
            block_ids=block_ids, place_ids=place_ids, voter_ids=voter_ids)

    case Problem_Types.group_canvas:
        try:
            result = db.get_dict(DEPOT, NODE_COORDS_DB_IDX)
        except KeyError:
            raise ValueError(
                "Depot point provided must be the id of point in the database's nodes"
            )

        depot = Point(
            lat=result["lat"],
            lon=result["lon"],
            type=NodeType.other,
            id="depot",
        )

        optimizer = GroupCanvas(
            block_ids=block_ids, place_ids=place_ids, voter_ids=voter_ids)

    case _:
        print(colored("Invalid problem type", color="red"))
        sys.exit(1)


"-----------------------------------------------------------------------------------------"
"                                      Optimize                                           "
" Run the optimizer on the subset of the universe, providing a starting location for the "
" group canvas problem and nothing for the turf split problem                             "
"-----------------------------------------------------------------------------------------"

if not args.no_optimize:
    routes: list[list[Point]] = optimizer(debug=True, time_limit_s=60)

    # process_solution(
    #     routes=routes,
    #     optimizer_points=optimizer.points,
    #     place_ids=place_ids,
    #     mix_distances=mix_distances,
    # )

    if routes is None:
        print(colored("Failed to generate lists", color="red"))
        sys.exit()

    pickle.dump(optimizer.points, open(optimizer_points_pickle_file, "wb"))
    print(colored("Generated lists and pickled optimizer points", color="green"))

"-----------------------------------------------------------------------------------------"
"                                      Post-Process                                       "
" Eliminate inaccuracies in the solution by applying some walkability and visibility      "
" heuristics (combining duplicate blocks, eliminating backtracking, etc.)                 "
" Also, generate the walk list files and visualizations                                   "
"-----------------------------------------------------------------------------------------"

# Load the optimizer points from pickle
optimizer_points = pickle.load(open(optimizer_points_pickle_file, "rb"))

# Process the solution
# process_solution(
#     routes=routes,
#     optimizer_points=optimizer_points,
#     place_ids=place_ids,
#     mix_distances=mix_distances,
# )
