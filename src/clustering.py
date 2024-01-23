"""
Cluster a large area into smaller areas suitable for turf split optimization.
"""

from __future__ import annotations

import sys
import numpy as np

from sklearn.cluster import BisectingKMeans, KMeans
from termcolor import colored

from src.config import (
    SUPER_CLUSTER_NUM_HOUSES,
    ABODE_DB_IDX,
    InternalPoint,
    BLOCK_DB_IDX,
)
from src.utils.db import Database
from src.utils.gps import pt_to_utm
from src.utils.viz import display_clustered_points
from src.distances.blocks import BlockDistances
from src.distances.nodes import NodeDistances


class Clustering:
    def __init__(self, block_ids: set[str], abode_ids: set[str]):
        self.db = Database()
        self.block_ids = block_ids
        self.abode_ids = abode_ids

    def calculate_tightness(self, block_ids: set[str], abode_ids: set[str]) -> float:
        """
        Calculate the tightness of a cluster.

        Args:
            block_ids (set[str]): The block ids in the cluster.
            abode_ids (set[str]): The abode ids in the cluster.

        Returns:
            float: The tightness of the cluster.
        """
        node_distances = NodeDistances(block_ids=block_ids, skip_update=True)

        block_distances = BlockDistances(
            block_ids=block_ids, node_distances=node_distances, skip_update=True
        )

        distance_matrix = block_distances.get_distance_matrix(block_ids=block_ids)

        return len(abode_ids) / (np.median(distance_matrix) / 1000)

    def cluster_blockwise(self) -> list[dict[str, set[str] | float]]:
        """
        Cluster a large area into smaller areas suitable for turf split optimization.

        Returns
        -------
            list[dict[str, set[str] | float]]: The clusters of blocks and abodes, and attributes.
                "block_ids" -> set of block ids in the cluster
                "abode_ids" -> set of abode ids in the cluster
                "tightness" -> the tightness of the cluster
        """
        node_distances = NodeDistances(block_ids=self.block_ids, skip_update=True)

        block_distances = BlockDistances(
            block_ids=self.block_ids, node_distances=node_distances, skip_update=True
        )

        distance_matrix = block_distances.get_distance_matrix(block_ids=self.block_ids)

        num_clusters = len(self.abode_ids) // SUPER_CLUSTER_NUM_HOUSES

        clustered = KMeans(n_clusters=num_clusters, random_state=0, tol=1e-9, n_init=10).fit(
            distance_matrix
        )

        block_id_clusters: list[set[str]] = [set() for _ in range(num_clusters)]

        for i, block_id in enumerate(self.block_ids):
            block_id_clusters[clustered.labels_[i]].add(block_id)

        abode_id_clusters: list[set[str]] = [set() for _ in range(num_clusters)]
        abode_clusters: list[int] = []
        abode_points: list[InternalPoint] = []

        for i, block_id in enumerate(self.block_ids):
            block_data = self.db.get_dict(block_id, BLOCK_DB_IDX)
            if block_data is None:
                print(
                    colored(
                        "Block {} not found in database".format(block_id), color="red"
                    )
                )
                sys.exit(1)

            for abode_id in block_data["abodes"]:
                if abode_id not in self.abode_ids:
                    continue
                abode = InternalPoint(
                    lat=block_data["abodes"][abode_id]["point"]["lat"],
                    lon=block_data["abodes"][abode_id]["point"]["lon"],
                )
                abode_points.append(abode)
                abode_clusters.append(clustered.labels_[i])

            abode_id_clusters[clustered.labels_[i]].update(
                set(block_data["abodes"].keys()).intersection(self.abode_ids)
            )

        display_clustered_points(abode_points, abode_clusters)

        tightnesses = [
            self.calculate_tightness(block_id_clusters[i], abode_id_clusters[i])
            for i in range(num_clusters)
        ]

        clusters = []
        for i in range(num_clusters):
            clusters.append(
                {
                    "block_ids": block_id_clusters[i],
                    "abode_ids": abode_id_clusters[i],
                    "tightness": tightnesses[i],
                }
            )

        return clusters

    def cluster_pointwise(self) -> list[dict[str, set[str] | float]]:
        """
        Cluster a large area into smaller areas suitable for turf split optimization.

        Returns
        -------
            list[dict[str, set[str] | float]]: The clusters of blocks and abodes, and attributes.
                "block_ids" -> set of block ids in the cluster
                "abode_ids" -> set of abode ids in the cluster
                "tightness" -> the tightness of the cluster

        Notes
        -----
            This method is not currently used (and may not be working), but is left here for future reference.
        """
        # Get point locations in UTM
        points: list[tuple[float, float]] = []

        # Convert the abode ids to a list for consistent ordering and indexing
        abode_ids_list = list(self.abode_ids)
        abode_points: list[InternalPoint] = []

        running_utm_zone_and_letter: tuple[int, str] = None

        for abode_id in abode_ids_list:
            abode = self.db.get_dict(abode_id, ABODE_DB_IDX)
            if abode is None:
                print(
                    colored(
                        "Abode {} not found in database".format(abode_id), color="red"
                    )
                )
                sys.exit(1)

            # Get the latitude and longitude from the block data
            block_data = self.db.get_dict(abode["block_id"], BLOCK_DB_IDX)
            if block_data is None:
                print(
                    colored(
                        "Block {} not found in database".format(abode["block_id"]),
                        color="red",
                    )
                )
                sys.exit(1)

            abode_point: InternalPoint = InternalPoint(
                lat=block_data["abodes"][abode_id]["lat"],
                lon=block_data["abodes"][abode_id]["lon"],
            )
            abode_points.append(abode_point)

            x, y, zone, let = pt_to_utm(abode_point)

            if running_utm_zone_and_letter is None:
                running_utm_zone_and_letter = (zone, let)
            elif running_utm_zone_and_letter != (zone, let):
                raise NotImplementedError("Cannot cluster across UTM zones")

            points.append([x, y])

        # Run spectral clustering with euclidian distance
        num_clusters = len(abode_ids_list) // SUPER_CLUSTER_NUM_HOUSES
        clustering = BisectingKMeans(n_init=10, n_clusters=num_clusters, random_state=0).fit(
            points
        )

        display_clustered_points(abode_points, clustering.labels_)

        abode_id_clusters: list[set[str]] = [set() for _ in range(num_clusters)]
        for i, abode_id in enumerate(abode_ids_list):
            abode_id_clusters[clustering.labels_[i]].add(abode_id)

        block_id_clusters: list[set[str]] = [set() for _ in range(num_clusters)]

        # For each block, assign it to the cluster which has the most houses on the block
        for block in self.block_ids:
            block_data = self.db.get_dict(block, BLOCK_DB_IDX)
            if block_data is None:
                print(
                    colored("Block {} not found in database".format(block), color="red")
                )
                sys.exit(1)

            num_houses_per_cluster = [
                set(block_data["abodes"].keys()).intersection(abode_id_clusters[i])
                for i in range(num_clusters)
            ]

            max_cluster = np.argmax(num_houses_per_cluster)

            block_id_clusters[max_cluster].add(block)

        tightnesses = [
            self.calculate_tightness(block_id_clusters[i], abode_id_clusters[i])
            for i in range(num_clusters)
        ]

        clusters = []
        for i in range(num_clusters):
            clusters.append(
                {
                    "block_ids": block_id_clusters[i],
                    "abode_ids": abode_id_clusters[i],
                    "tightness": tightnesses[i],
                }
            )

        return clusters

    def __call__(self) -> list[dict[str, set[str] | float]]:
        """
        Cluster a large area into smaller areas suitable for turf split optimization.

        Returns
        -------
            list[dict[str, set[str] | float]]: The clusters of blocks and abodes, and attributes.
                "block_ids" -> set of block ids in the cluster
                "abode_ids" -> set of abode ids in the cluster
                "tightness" -> the tightness of the cluster
        """
        return self.cluster_blockwise()
