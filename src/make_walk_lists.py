# TODO: Fix arbitrary large distances throughout package

from __future__ import annotations

import json
import os
import pickle
import sys
from copy import deepcopy

from kmedoids import KMedoids
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from termcolor import colored

from src.config import (
    AREA_ID,
    BASE_DIR,
    DEPOT,
    GROUP_CANVAS_FULL,
    KEEP_APARTMENTS,
    NUM_LISTS,
    TURF_SPLIT,
    NodeType,
    Point,
    Solution,
    blocks_file_t,
    node_coords_file,
    optimizer_points_pickle_file,
    pt_id,
    requested_blocks_file,
    solution_path,
)
from src.distances.blocks import BlockDistances
from src.distances.houses import HouseDistances
from src.distances.mix import MixDistances
from src.distances.nodes import NodeDistances
from src.optimize import Optimizer
from src.post_process import process_solution
from src.viz_utils import display_blocks, display_clustered_blocks

if len(sys.argv) == 2 and sys.argv[1] == "-no":
    NO_OPTIMIZE = True
else:
    NO_OPTIMIZE = False


# Load the requested blocks
requested_blocks: blocks_file_t = json.load(open(requested_blocks_file))

display_blocks(requested_blocks)[0].save(os.path.join(BASE_DIR, "viz", "segments.html"))

# Generate node distance matrix
NodeDistances(requested_blocks)

# Generate block distance matrix
BlockDistances(requested_blocks)

# Initialize calculator for mixed distances
MixDistances()

"-------------------------------------------------------------------------------------------------------"
"                                      Cluster                                                          "
" Using a clustering algorithm, we cluster the blocks and designate a center node for each cluster      "
" Clusters are used for partitioning the space into more reasonable and optimizable areas               "
"-------------------------------------------------------------------------------------------------------"
# region Cluster
# Cluster blocks using kmedoids
distance_matrix = BlockDistances.get_distance_matrix()
# db = DBSCAN(metric="precomputed", eps=400, min_samples=10).fit(distance_matrix)
# db = KMedoids(n_clusters=25, metric="precomputed").fit(distance_matrix)
db = AgglomerativeClustering(
    n_clusters=None, linkage="complete", distance_threshold=1000, metric="precomputed"
).fit(distance_matrix)
labels: list[int] = db.labels_  # type: ignore

# Expand labels into a list of block groups
clustered_blocks: list[blocks_file_t] = [
    {
        b_id: b_info
        for i, (b_id, b_info) in enumerate(requested_blocks.items())
        if labels[i] == k
    }  # Blocks in cluster k
    for k in range(max(labels))
]


def cluster_to_houses(cluster: blocks_file_t) -> list[Point]:
    """Convert a list of blocks to its corresponding list of houses."""
    points: list[Point] = []

    for block in cluster.values():
        # Duplicate addresses from apartments may occur. For now, only insert once

        for house_id, house_data in block["addresses"].items():
            # TODO: Move this conditional outward so we can get rid of this whole method
            # if not KEEP_APARTMENTS and " APT " in house_data["display_address"]:
            #     continue

            # if "3245 BEECHWOOD BLVD" in house_data["display_address"]:
            #     # This is the only address where there are multiple houses on multiple blocks
            #     continue

            # TODO: Maybe do this earlier to get rid of this method
            points.append(
                Point(
                    lat=house_data["lat"],
                    lon=house_data["lon"],
                    id=house_id,
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


clustered_points: list[list[Point]] = [cluster_to_houses(c) for c in clustered_blocks]

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

display_clustered_blocks(requested_blocks, labels, centers).save(
    os.path.join(BASE_DIR, "viz", "clusters.html")
)
# endregion

# Use all blocks as a single area
# areas = [i for i in range(len(clustered_blocks))]

areas = [8]

# areas = [1]

area = clustered_points[areas[0]]
area_blocks = deepcopy(clustered_blocks[areas[0]])
for i in range(1, len(areas)):
    area += clustered_points[areas[i]]
    area_blocks.update(clustered_blocks[areas[i]])

depot: Point | list[Point] = []

if TURF_SPLIT:
    # Create a depot at each intersection
    depot = []
    unique_intersection_ids: set[str] = set()
    for block in area_blocks.values():
        if pt_id(block["nodes"][0]) not in unique_intersection_ids:
            new_pt = Point(
                lat=block["nodes"][0]["lat"],
                lon=block["nodes"][0]["lon"],
                type=NodeType.node,
                id=pt_id(block["nodes"][0]),
            )
            depot.append(new_pt)
            unique_intersection_ids.add(pt_id(new_pt))
        if pt_id(block["nodes"][-1]) not in unique_intersection_ids:
            new_pt = Point(
                lat=block["nodes"][-1]["lat"],
                lon=block["nodes"][-1]["lon"],
                type=NodeType.node,
                id=pt_id(block["nodes"][-1]),
            )
            depot.append(new_pt)
            unique_intersection_ids.add(pt_id(new_pt))

    # Generate the house distance matrix
    HouseDistances(area_blocks)
elif GROUP_CANVAS_FULL:
    # depot = deepcopy(centers[areas[0]])
    # depot["id"] = "depot"
    # depot["type"] = NodeType.other

    # Generate house distance matrix, and distances to the depot
    HouseDistances(requested_blocks)
else:
    node_coords_file = json.load(open(node_coords_file))

    try:
        result = node_coords_file[DEPOT]
    except KeyError:
        print(colored("Depot not found in node_coords_file", color="red"))
        sys.exit()

    depot = Point(
        lat=result["lat"],
        lon=result["lon"],
        type=NodeType.other,
        id="depot",
    )

    # Generate house distance matrix, and distances to the depot
    HouseDistances(area_blocks, depot)

"-----------------------------------------------------------------------------------------"
"                                      Optimize                                           "
" Run the optimizer on the subset of the universe, providing a starting location for the "
" group canvas problem and nothing for the turf split problem                             "
"-----------------------------------------------------------------------------------------"

# TODO: Run the optimizer for each cluster, saving the lists into separate folders
# Then, we can re-run on individual clusters and such if we need (or do more logic)

if GROUP_CANVAS_FULL:
    for i, (cluster, center) in enumerate(zip(clustered_points, centers)):
        print(colored("Optimizing cluster {}".format(i), color="green"))
        save_path = os.path.join(BASE_DIR, "regions", AREA_ID, "areas", str(i))
        os.makedirs(save_path, exist_ok=True)

        viz_dir = os.path.join(save_path, "viz")
        problem_dir = os.path.join(save_path, "problem")
        os.makedirs(viz_dir, exist_ok=True)
        os.makedirs(problem_dir, exist_ok=True)

        optimizer = Optimizer(cluster, starting_locations=center, save_path=problem_dir)
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
            requested_blocks=requested_blocks,
            viz_path=viz_dir,
            problem_path=problem_dir
        )

        # TEMPORARY NOTE
        # sys.exit(0)

    sys.exit(0)

if not NO_OPTIMIZE:
    optimizer = Optimizer(area, num_lists=NUM_LISTS, starting_locations=depot)
    optimizer.optimize()
    solution = optimizer.process_solution(solution_path)

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
solution: Solution = Optimizer.process_solution(solution_path)

# Process the solution
process_solution(
    solution=solution,
    optimizer_points=optimizer_points,
    requested_blocks=requested_blocks,
)
