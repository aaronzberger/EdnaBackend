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
from src.post_processing.post_process import process_solution
from src.utils.db import Database
from src.utils.viz import display_clustered_blocks

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

# Generate node distance matrix
node_distances = NodeDistances(block_ids=block_ids, skip_update=True)

# Generate block distance matrix
block_distances = BlockDistances(
    block_ids=block_ids, node_distances=node_distances, skip_update=True
)


# Define some util methods for post-processing the clusters back into optimization-able format
def cluster_to_places(cluster: blocks_file_t) -> list[Point]:
    """
    Convert a list of blocks to a list of geographic points of its houses.

    Parameters
    ----------
    blocks_file_t

    Returns
    -------
    list[Point]
    """
    points: list[Point] = []

    for block in cluster.values():
        # Duplicate addresses from apartments may occur. For now, only insert once

        for place_id, place_data in block["places"].items():
            if place_id not in place_ids:
                # This abode is on the global block, but isn't in this universe.
                # It likely doesn't have any voters in this campaign's universe.
                continue

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
    """
    Convert a list of blocks to a list of geographic points of its intersections.

    Parameters
    ----------
    blocks_file_t

    Returns
    -------
    list[Point]
    """
    points: list[Point] = []
    inserted_point_ids: set[str] = set()

    for block in cluster.values():
        for i in [0, -1]:
            if pt_id(block["nodes"][i]) not in inserted_point_ids:
                points.append(
                    Point(
                        lat=block["nodes"][i]["lat"],
                        lon=block["nodes"][i]["lon"],
                        id=pt_id(block["nodes"][i]),
                        type=NodeType.node,
                    )
                )

    return points


"-------------------------------------------------------------------------------------------------------"
"                                       Super-Clustering                                                "
" Cluster the blocks to partition the space into more optimizable areas                                 "
" This simply divides the blocks, places, and voters into clusters which are then independently solved  "
"-------------------------------------------------------------------------------------------------------"
# region Cluster
if not SUPER_CLUSTERING:
    # Assume one large cluster
    block_id_clusters = [block_ids]
    place_id_clusters = [place_ids]
    voter_id_clusters = [voter_ids]
else:
    distance_matrix = block_distances.get_distance_matrix(block_ids=block_ids)

    if os.path.exists(clustering_pickle_file):
        clustered = pickle.load(open(clustering_pickle_file, "rb"))
        clustered_points = clustered["clustered_points"]
        clustered_blocks = clustered["clustered_blocks"]
        centers = clustered["centers"]
    else:
        match PROBLEM_TYPE:
            case Problem_Types.turf_split:
                clustered = DBSCAN(metric="precomputed", eps=400, min_samples=10).fit(
                    distance_matrix
                )
            case Problem_Types.completed_group_canvas | Problem_Types.group_canvas:
                clustered = AgglomerativeClustering(
                    n_clusters=None,
                    linkage="complete",
                    distance_threshold=1000,
                    metric="precomputed",
                ).fit(distance_matrix)
            case _:
                print(
                    colored(
                        "Invalid problem type for clustering. Exiting...", color="red"
                    )
                )
                sys.exit(1)

        labels: list[int] = clustered.labels_

        # Expand labels into a list of block groups
        clustered_blocks: list[blocks_file_t] = []

        for k in range(max(labels)):
            k_dict = {}
            for label, block_id in zip(labels, block_ids):
                if label == k:
                    k_dict[block_id] = db.get_dict(block_id, BLOCK_DB_IDX)
            clustered_blocks.append(k_dict)

        clustered_points: list[list[Point]] = [
            cluster_to_places(c) for c in clustered_blocks
        ]

        print("Total of {} points".format(sum([len(c) for c in clustered_points])))

        # Print the clusters with too few houses
        for i, cluster in enumerate(clustered_points):
            if len(cluster) < 10:
                print(
                    colored(
                        "Cluster {}s has only {} houses".format(i, len(cluster)),
                        color="red",
                    )
                )

        centers: list[Point] = []
        print("Getting centers for {} clusters".format(len(clustered_blocks)))
        node_distance_snapshot = node_distances.snapshot()
        for cluster in clustered_blocks:
            insertections: list[Point] = cluster_to_intersections(cluster)

            # Choose the intersection which minimizes the sum of distances to all other intersections
            min_sum = float("inf")
            min_intersection = None
            for intersection in insertections:
                sum = 0
                for other_intersection in insertections:
                    distance = node_distance_snapshot.get_distance(
                        intersection, other_intersection
                    )
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
        del node_distance_snapshot

        display_clustered_blocks(list(block_ids), labels, centers)

        pickle.dump(
            {
                "clustered_points": clustered_points,
                "clustered_blocks": clustered_blocks,
                "centers": centers,
            },
            open(clustering_pickle_file, "wb"),
        )
        # assert len(block_id_clusters) == len(place_id_clusters) == len(voter_id_clusters)
# endregion

"-------------------------------------------------------------------------------------------------------"
"                                               Building                                                "
" Build the depots and houses for the problem and areas specified                                       "
"-------------------------------------------------------------------------------------------------------"

for block_id_cluster, place_id_cluster, voter_id_cluster in zip(
    block_id_clusters, place_id_clusters, voter_id_clusters
):
    # Convert the places into points for optimization
    abode_points = cluster_to_places(block_id_cluster)

    optimizer: Optional[Optimizer] = None

    match PROBLEM_TYPE:
        case Problem_Types.turf_split:
            depots: list[Point] = cluster_to_intersections(block_id_cluster)

            # TODO: We're not providing depots here. Is it calling live? If so, provide them here (edit init method of HouseDistances)
            house_distances = HouseDistances(
                block_ids=block_id_cluster, node_distances=node_distances, depots=depots
            )

            mix_distances = MixDistances(
                node_distances=node_distances, house_distances=house_distances
            )

            optimizer = TurfSplit(
                houses=abode_points,
                potential_depots=depots,
                num_routes=NUM_LISTS,
                mix_distances=mix_distances,
            )

        case Problem_Types.group_canvas:
            depot_points = []
            for depot in DEPOTS:
                try:
                    result = db.get_dict(depot, NODE_COORDS_DB_IDX)
                except KeyError:
                    raise ValueError(
                        "Each depot point provided must be the id of point in the database's nodes"
                    )

                depot_points.append(
                    Point(
                        lat=result["lat"],
                        lon=result["lon"],
                        type=NodeType.other,
                        id="depot",
                    )
                )

            house_distances = HouseDistances(
                block_ids=block_id_cluster,
                node_distances=node_distances,
                depots=depot_points,
            )

            mix_distances = MixDistances(
                node_distances=node_distances, house_distances=house_distances
            )

            optimizer = GroupCanvas(
                houses=abode_points, depots=depot_points, mix_distances=mix_distances
            )
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

        process_solution(
            solution=solution,
            optimizer_points=optimizer.points,
            place_ids=place_ids,
            mix_distances=mix_distances,
        )

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
    # solution: Solution = Optimizer.process_solution(default_solution_path)

    # Process the solution
    process_solution(
        solution=solution,
        optimizer_points=optimizer_points,
        place_ids=place_ids,
        mix_distances=mix_distances,
    )
