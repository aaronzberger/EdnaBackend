"""
Cluster a large area into smaller areas suitable for turf split optimization.
"""

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
    DEPOT,
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


class Clustering:
    def __init__(self, block_ids: set[str], place_ids: set[str], voter_ids: set[str]):
        self.db = Database()
        self.block_ids = block_ids
        self.place_ids = place_ids
        self.voter_ids = voter_ids

    def cluster_to_places(self, cluster: blocks_file_t) -> list[Point]:
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

        for block in cluster:
            block = self.db.get_dict(block, BLOCK_DB_IDX)
            # Duplicate addresses from apartments may occur. For now, only insert once

            for place_id, place_data in block["places"].items():
                if place_id not in self.place_ids:
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

    def cluster_to_intersections(self, cluster: blocks_file_t) -> list[Point]:
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

        for block in cluster:
            block = self.db.get_dict(block, BLOCK_DB_IDX)
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
                    inserted_point_ids.add(pt_id(block["nodes"][i]))

        return points

    def __call__(self) -> dict[str, list[set[str]]]:
        """
        Cluster a large area into smaller areas suitable for turf split optimization.

        Args:
            block_ids (set[str]): The block ids to cluster.
            place_ids (set[str]): The place ids to cluster.
            voter_ids (set[str]): The voter ids to cluster.

        Returns:
            dict[str, list[set[str]]]: The clusters.
        """
        raise NotImplementedError


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
