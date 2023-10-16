"""Run the optimizer and post-process the solution to generate walk lists."""
# TODO: Fix arbitrary large distances throughout package

from __future__ import annotations

import json
import os
import pickle
import sys
from copy import deepcopy
from typing import Optional

from sklearn.cluster import DBSCAN, AgglomerativeClustering
from termcolor import colored
import argparse

from src.config import (
    BLOCK_DB_IDX,
    CAMPAIGN_NAME,
    BASE_DIR,
    DEPOT,
    PLACE_DB_IDX,
    PROBLEM_TYPE,
    NODE_COORDS_DB_IDX,
    VOTER_DB_IDX,
    Problem_Types,
    CAMPAIGN_SUBSET_DB_IDX,
    NUM_LISTS,
    NodeType,
    Point,
    Solution,
    blocks_file_t,
    node_coords_file,
    optimizer_points_pickle_file,
    pt_id,
    default_solution_path,
    clustering_pickle_file
)
from src.distances.blocks import BlockDistances
from src.distances.houses import HouseDistances
from src.distances.mix import MixDistances
from src.distances.nodes import NodeDistances
from src.optimize.completed_group_canvas import CompletedGroupCanvas
from src.optimize.group_canvas import GroupCanvas
from src.optimize.optimizer import Optimizer
from src.optimize.turf_split import TurfSplit
from src.post_processing.post_process import process_solution
from src.utils.viz import display_clustered_blocks
from src.utils.db import Database


parser = argparse.ArgumentParser(
    prog="make_walk_lists.py",
    description="Generate walk lists",
    epilog="Developed by VoteFalcon")
parser.add_argument("-n", "--no-optimize", action="store_true", help="Skip the optimization step, post-process only")
parser.add_argument("-r", "--restart", action="store_true", help="Force-perform the optimization on all clusters")
args = parser.parse_args()

db = Database()

# Walk up the database from voters to places to blocks
# To retrieve all nodes, take the first and last node from all blocks with places which have voters in this campaign
voter_ids = db.get_set(CAMPAIGN_NAME, CAMPAIGN_SUBSET_DB_IDX)

print(f'Found {len(voter_ids)} voters')

place_ids: set[str] = set()
for voter in voter_ids:
    place = db.get_dict(voter, VOTER_DB_IDX)
    if place is None:
        print(colored("Voter {} not found in database".format(voter), color="red"))
        sys.exit(1)
    place_ids.add(place["place"])

print('Found {} places'.format(len(place_ids)))

block_ids: set[str] = set()
for place in place_ids:
    place_data = db.get_dict(place, PLACE_DB_IDX)
    if place_data is None:
        print(colored("Place {} not found in database".format(place), color="red"))
        sys.exit(1)
    block_ids.add(place_data["block_id"])

print('Found {} blocks'.format(len(block_ids)))

# Generate node distance matrix
NodeDistances(block_ids=block_ids, skip_update=True)

# Generate block distance matrix
BlockDistances(block_ids=block_ids, skip_update=True)

# Initialize calculator for mixed distances
MixDistances()

"-------------------------------------------------------------------------------------------------------"
"                                      Cluster                                                          "
" Using a clustering algorithm, we cluster the blocks and designate a center node for each cluster      "
" Clusters are used for partitioning the space into more reasonable and optimizable areas               "
"-------------------------------------------------------------------------------------------------------"
# region Cluster
distance_matrix = BlockDistances.get_distance_matrix(block_ids=block_ids)

if os.path.exists(clustering_pickle_file):
    clustered = pickle.load(open(clustering_pickle_file, "rb"))
    clustered_points = clustered["clustered_points"]
    clustered_blocks = clustered["clustered_blocks"]
    centers = clustered["centers"]
else:
    match PROBLEM_TYPE:
        case Problem_Types.turf_split:
            clustered = DBSCAN(metric="precomputed", eps=400, min_samples=10).fit(distance_matrix)
        case Problem_Types.completed_group_canvas:
            clustered = AgglomerativeClustering(
                n_clusters=None, linkage="complete", distance_threshold=1000, metric="precomputed"
            ).fit(distance_matrix)
        case _:
            print(colored("Invalid problem type", color="red"))
            sys.exit(1)

    labels: list[int] = clustered.labels_  # type: ignore

    # Expand labels into a list of block groups
    clustered_blocks: list[blocks_file_t] = []

    for k in range(max(labels)):
        k_dict = {}
        for label, block_id in zip(labels, block_ids):
            if label == k:
                k_dict[block_id] = db.get_dict(block_id, BLOCK_DB_IDX)
        clustered_blocks.append(k_dict)

    def cluster_to_places(cluster: blocks_file_t) -> list[Point]:
        """Convert a list of blocks to its corresponding list of houses."""
        points: list[Point] = []

        for block in cluster.values():
            # Duplicate addresses from apartments may occur. For now, only insert once

            for place_id, place_data in block["places"].items():
                # TODO: Maybe do this earlier to get rid of this method
                points.append(
                    Point(
                        lat=place_data["lat"],
                        lon=place_data["lon"],
                        id=place_id,
                        type=NodeType.house,
                    )
                )

        return points

    def cluster_to_intersections(cluster: blocks_file_t) -> list[Point]:
        """Convert a list of blocks to its corresponding list of intersections."""
        points: list[Point] = []

        for block in cluster.values():
            # Duplicate addresses from apartments may occur. For now, only insert once

            for node in block["nodes"]:
                points.append(
                    Point(
                        lat=node["lat"],
                        lon=node["lon"],
                        id=pt_id(node),
                        type=NodeType.node,
                    )
                )

        return points

    clustered_points: list[list[Point]] = [cluster_to_places(c) for c in clustered_blocks]

    # Print the clusters with too few houses
    for i, cluster in enumerate(clustered_points):
        if len(cluster) < 10:
            print(colored("Cluster {} has only {} houses".format(i, len(cluster)), color="red"))

    centers: list[Point] = []
    for cluster in clustered_blocks:
        insertections: list[Point] = cluster_to_intersections(cluster)

        # Choose the intersection which minimizes the sum of distances to all other intersections
        min_sum = float("inf")
        min_intersection = None
        for intersection in insertections:
            sum = 0
            for other_intersection in insertections:
                distance = NodeDistances.get_distance(intersection, other_intersection)
                if distance is None:
                    distance = 1600
                sum += distance
            if sum < min_sum:
                min_sum = sum
                min_intersection = intersection

        if min_intersection is None:
            print(colored("Intersection not found", color="red"))
            sys.exit()

        centers.append(min_intersection)

    display_clustered_blocks(list(block_ids), labels, centers)

    pickle.dump(
        {
            "clustered_points": clustered_points,
            "clustered_blocks": clustered_blocks,
            "centers": centers,
        },
        open(clustering_pickle_file, "wb"),
        )
# endregion

"-------------------------------------------------------------------------------------------------------"
"                                               Building                                                "
" Build the depots and houses for the problem and areas specified                                       "
"-------------------------------------------------------------------------------------------------------"

# Use all blocks as a single area
# areas = [i for i in range(len(clustered_blocks))]

areas = [44]
area = clustered_points[areas[0]]
area_blocks = deepcopy(clustered_blocks[areas[0]])
for i in range(1, len(areas)):
    area += clustered_points[areas[i]]
    area_blocks.update(clustered_blocks[areas[i]])

optimizer: Optional[Optimizer] = None

match PROBLEM_TYPE:
    case Problem_Types.turf_split:
        # Create a depot at each intersection
        depots = []
        unique_intersection_ids: set[str] = set()
        for block in area_blocks.values():
            if pt_id(block["nodes"][0]) not in unique_intersection_ids:
                new_pt = Point(
                    lat=block["nodes"][0]["lat"],
                    lon=block["nodes"][0]["lon"],
                    type=NodeType.node,
                    id=pt_id(block["nodes"][0]),
                )
                depots.append(new_pt)
                unique_intersection_ids.add(pt_id(new_pt))
            if pt_id(block["nodes"][-1]) not in unique_intersection_ids:
                new_pt = Point(
                    lat=block["nodes"][-1]["lat"],
                    lon=block["nodes"][-1]["lon"],
                    type=NodeType.node,
                    id=pt_id(block["nodes"][-1]),
                )
                depots.append(new_pt)
                unique_intersection_ids.add(pt_id(new_pt))

        # Generate the house distance matrix
        HouseDistances(block_ids=list(area_blocks.keys()))

        # Create the optimizer
        optimizer = TurfSplit(houses=area, depots=depots, num_lists=NUM_LISTS)

    case Problem_Types.group_canvas:
        try:
            result = db.get_dict(DEPOT, NODE_COORDS_DB_IDX)
        except KeyError:
            print(colored("Depot not found in node_coords_file", color="red"))
            sys.exit()

        depot = Point(
            lat=result["lat"],
            lon=result["lon"],
            type=NodeType.other,
            id="depot",
        )
        HouseDistances(block_ids=list(area_blocks.keys()), depot=depot)

        optimizer = GroupCanvas(houses=area, depot=depot, num_lists=NUM_LISTS)
        optimizer.build_problem()

    case Problem_Types.completed_group_canvas:
        HouseDistances(block_ids=list(area_blocks.keys()))

        # The problem is created and ran at "optimize"-time


"-----------------------------------------------------------------------------------------"
"                                      Optimize                                           "
" Run the optimizer on the subset of the universe, providing a starting location for the "
" group canvas problem and nothing for the turf split problem                             "
"-----------------------------------------------------------------------------------------"


match PROBLEM_TYPE:
    case Problem_Types.completed_group_canvas:
        for i, (cluster, center) in enumerate(zip(clustered_points, centers)):
            if not args.restart and os.path.exists(os.path.join(BASE_DIR, "regions", CAMPAIGN_NAME, "areas", str(i))):
                continue
            print(colored("Optimizing cluster {}".format(i), color="green"))
            save_path = os.path.join(BASE_DIR, "regions", CAMPAIGN_NAME, "areas", str(i))

            # Remove the old files (and all subdirectories)
            if os.path.isdir(save_path):
                os.system("rm -rf {}".format(save_path))

            os.makedirs(save_path, exist_ok=True)

            viz_dir = os.path.join(save_path, "viz")
            problem_dir = os.path.join(save_path, "problem")
            os.makedirs(viz_dir, exist_ok=True)
            os.makedirs(problem_dir, exist_ok=True)

            optimizer = CompletedGroupCanvas(houses=cluster, depot=center)
            optimizer.optimize()

            pickle.dump(optimizer.points, open(os.path.join(problem_dir, "points.pkl"), "wb"))

            solution = optimizer.process_solution(
                os.path.join(problem_dir, "solution.json")
            )

            if solution is None:
                print(
                    colored(
                        "Failed to generate lists for cluster {}".format(cluster),
                        color="red",
                    )
                )
                continue

            process_solution(
                solution=solution,
                optimizer_points=optimizer.points,
                viz_path=viz_dir,
                problem_path=problem_dir,
                id=str(i)
            )
    case _:
        if optimizer is None:
            print(colored("Invalid problem type", color="red"))
            sys.exit(1)
        if not args.no_optimize:
            optimizer.optimize()
            solution = optimizer.process_solution(default_solution_path)

            if solution is None:
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

# Load the solution file
solution: Solution = Optimizer.process_solution(default_solution_path)

# Process the solution
process_solution(
    solution=solution,
    optimizer_points=optimizer_points,
)
