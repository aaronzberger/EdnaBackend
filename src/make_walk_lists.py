# TODO: Fix arbitrary large distances throughout package

from __future__ import annotations

import json
import os
import pickle
import sys
from copy import deepcopy

from sklearn.cluster import DBSCAN
from termcolor import colored

from gps_utils import SubBlock
from src.config import (
    BASE_DIR,
    DEPOT,
    KEEP_APARTMENTS,
    NUM_LISTS,
    TURF_SPLIT,
    NodeType,
    Point,
    blocks_file_t,
    pt_id,
    requested_blocks_file
)
from src.distances.blocks import BlockDistances
from src.distances.houses import HouseDistances
from src.distances.mix import MixDistances
from src.distances.nodes import NodeDistances
from src.optimize import Optimizer
from src.post_process import PostProcess
from src.viz_utils import (
    display_blocks,
    display_clustered_blocks,
    display_individual_walk_lists,
    display_walk_lists,
)

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
db = DBSCAN(metric="precomputed", eps=400, min_samples=10).fit(distance_matrix)
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
            if not KEEP_APARTMENTS and " APT " in house_data["display_address"]:
                continue

            if "3245 BEECHWOOD BLVD" in house_data["display_address"]:
                # This is the only address where there are multiple houses on multiple blocks
                continue

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


clustered_points: list[list[Point]] = [cluster_to_houses(c) for c in clustered_blocks]

centers = [c[0] for c in clustered_points]
display_clustered_blocks(requested_blocks, labels, centers).save(
    os.path.join(BASE_DIR, "viz", "clusters.html")
)
# endregion

areas = [1]
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
else:
    depot = DEPOT

    # Generate house distance matrix, and distances to the depot
    HouseDistances(area_blocks, depot)

"-----------------------------------------------------------------------------------------"
"                                      Optimize                                           "
" Run the optimizer on the subset of the universe, providing a startting location for the "
" group canvas problem and nothing for the turf split problem                             "
"-----------------------------------------------------------------------------------------"
# region Optimize
optimizer = Optimizer(area, num_lists=NUM_LISTS, starting_locations=depot)
solution = optimizer.optimize()

if solution is None:
    print(colored("Failed to generate lists", color="red"))
    sys.exit()
# endregion

"-----------------------------------------------------------------------------------------"
"                                      Post-Process                                       "
" Eliminate innacuracies in the solution by applying some walkability and visibility      "
" heuristics (combining duplicate blocks, eliminating backtracking, etc.)                 "
" Also, generate the walk list files and visualizations                                   "
"-----------------------------------------------------------------------------------------"
# region: Post-Process

pickle.dump(
    optimizer.points, open(os.path.join(BASE_DIR, "optimize", "points.pkl"), "wb")
)

post_processor = PostProcess(requested_blocks, points=optimizer.points)
walk_lists: list[list[SubBlock]] = []
for i, tour in enumerate(solution["tours"]):
    # Do not count the starting location service at the start or end
    tour["stops"] = tour["stops"][1:-1] if TURF_SPLIT else tour["stops"]

    if len(tour["stops"]) == 0:
        print("List {} has 0 stops".format(i))
        continue

    walk_lists.append(post_processor.post_process(tour))

# Save the walk lists
display_walk_lists(walk_lists).save(os.path.join(BASE_DIR, "viz", "walk_lists.html"))

list_visualizations = display_individual_walk_lists(walk_lists)
for i, walk_list in enumerate(list_visualizations):
    walk_list.save(os.path.join(BASE_DIR, "viz", "walk_lists", "{}.html".format(i)))

for i in range(len(walk_lists)):
    post_processor.generate_file(
        walk_lists[i], os.path.join(BASE_DIR, "viz", "files", f"{i}.json")
    )

# endregion
