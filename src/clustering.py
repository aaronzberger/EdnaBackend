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
    PLACE_DB_IDX,
    Point,
    BLOCK_DB_IDX
)
from src.utils.db import Database
from src.utils.gps import pt_to_utm
from src.utils.viz import display_clustered_points
from src.distances.blocks import BlockDistances
from src.distances.nodes import NodeDistances


class Clustering:
    def __init__(self, block_ids: set[str], place_ids: set[str]):
        self.db = Database()
        self.block_ids = block_ids
        self.place_ids = place_ids

    def calculate_tightness(self, block_ids: set[str], place_ids: set[str]) -> float:
        """
        Calculate the tightness of a cluster.

        Args:
            block_ids (set[str]): The block ids in the cluster.
            place_ids (set[str]): The place ids in the cluster.

        Returns:
            float: The tightness of the cluster.
        """
        node_distances = NodeDistances(
            block_ids=block_ids, skip_update=True
        )

        block_distances = BlockDistances(
            block_ids=block_ids, node_distances=node_distances, skip_update=True
        )

        distance_matrix = block_distances.get_distance_matrix(block_ids=block_ids)

        return len(place_ids) / (np.median(distance_matrix) / 1000)

    def cluster_blockwise(self) -> list[dict[str, set[str] | float]]:
        """
        Cluster a large area into smaller areas suitable for turf split optimization.

        Returns
        -------
            list[dict[str, set[str] | float]]: The clusters of blocks and places, and attributes.
                "block_ids" -> set of block ids in the cluster
                "place_ids" -> set of place ids in the cluster
                "tightness" -> the tightness of the cluster
        """
        node_distances = NodeDistances(
            block_ids=self.block_ids, skip_update=True
        )

        block_distances = BlockDistances(
            block_ids=self.block_ids, node_distances=node_distances, skip_update=True
        )

        distance_matrix = block_distances.get_distance_matrix(block_ids=self.block_ids)

        num_clusters = len(self.place_ids) // SUPER_CLUSTER_NUM_HOUSES

        clustered = KMeans(n_clusters=num_clusters, random_state=0, tol=1e-9).fit(distance_matrix)

        block_id_clusters: list[set[str]] = [set() for _ in range(num_clusters)]

        for i, block_id in enumerate(self.block_ids):
            block_id_clusters[clustered.labels_[i]].add(block_id)

        place_id_clusters: list[set[str]] = [set() for _ in range(num_clusters)]
        place_clusters: list[int] = []
        places: list[Point] = []

        for i, block_id in enumerate(self.block_ids):
            block_data = self.db.get_dict(block_id, BLOCK_DB_IDX)
            if block_data is None:
                print(colored("Block {} not found in database".format(block_id), color="red"))
                sys.exit(1)

            for place_id in block_data["places"]:
                if place_id not in self.place_ids:
                    continue
                place = Point(lat=block_data["places"][place_id]["lat"], lon=block_data["places"][place_id]["lon"])
                places.append(place)
                place_clusters.append(clustered.labels_[i])

            place_id_clusters[clustered.labels_[i]].update(set(block_data["places"].keys()).intersection(self.place_ids))

        display_clustered_points(places, place_clusters)

        tightnesses = [self.calculate_tightness(block_id_clusters[i], place_id_clusters[i]) for i in range(num_clusters)]

        clusters = []
        for i in range(num_clusters):
            clusters.append({
                "block_ids": block_id_clusters[i],
                "place_ids": place_id_clusters[i],
                "tightness": tightnesses[i]
            })

        return clusters

    def cluster_pointwise(self) -> list[dict[str, set[str] | float]]:
        """
        Cluster a large area into smaller areas suitable for turf split optimization.

        Returns
        -------
            list[dict[str, set[str] | float]]: The clusters of blocks and places, and attributes.
                "block_ids" -> set of block ids in the cluster
                "place_ids" -> set of place ids in the cluster
                "tightness" -> the tightness of the cluster
        """
        # Get point locations in UTM
        points: list[tuple[float, float]] = []

        # Convert the place ids to a list for consistent ordering and indexing
        place_ids_list = list(self.place_ids)
        place_points: list[Point] = []

        running_utm_zone_and_letter: tuple[int, str] = None

        for place_id in place_ids_list:
            place = self.db.get_dict(place_id, PLACE_DB_IDX)
            if place is None:
                print(colored("Place {} not found in database".format(place_id), color="red"))
                sys.exit(1)

            # Get the latitude and longitude from the block data
            block_data = self.db.get_dict(place["block_id"], BLOCK_DB_IDX)
            if block_data is None:
                print(colored("Block {} not found in database".format(place["block_id"]), color="red"))
                sys.exit(1)

            place_point: Point = Point(lat=block_data["places"][place_id]["lat"], lon=block_data["places"][place_id]["lon"])
            place_points.append(place_point)

            x, y, zone, let = pt_to_utm(place_point)

            if running_utm_zone_and_letter is None:
                running_utm_zone_and_letter = (zone, let)
            elif running_utm_zone_and_letter != (zone, let):
                raise NotImplementedError("Cannot cluster across UTM zones")

            points.append([x, y])

        # Normalize the points
        # points = np.array(points)
        # points -= points.mean(axis=0)
        # points /= points.std(axis=0)

        print(f'Max x = {max(x for x, _ in points)}')
        print(f'Max y = {max(y for _, y in points)}')
        print(f'Min x = {min(x for x, _ in points)}')
        print(f'Min y = {min(y for _, y in points)}')

        print(f'Clustering {len(points)} points')

        # Run spectral clustering with euclidian distance
        num_clusters = len(place_ids_list) // SUPER_CLUSTER_NUM_HOUSES
        # clustering = Birch(
        #     threshold=1000, n_clusters=num_clusters).fit(points)
        clustering = BisectingKMeans(
            n_clusters=num_clusters, random_state=0).fit(points)

        display_clustered_points(place_points, clustering.labels_)

        for i in range(max(clustering.labels_) + 1):
            print(f'Cluster {i} has {sum(clustering.labels_ == i)} points')

        place_id_clusters: list[set[str]] = [set() for _ in range(num_clusters)]
        for i, place_id in enumerate(place_ids_list):
            place_id_clusters[clustering.labels_[i]].add(place_id)

        print("SIZES", [len(i) for i in place_id_clusters])

        block_id_clusters: list[set[str]] = [set() for _ in range(num_clusters)]

        # For each block, assign it to the cluster which has the most houses on the block
        for block in self.block_ids:
            block_data = self.db.get_dict(block, BLOCK_DB_IDX)
            if block_data is None:
                print(colored("Block {} not found in database".format(block), color="red"))
                sys.exit(1)

            num_houses_per_cluster = [set(block_data["places"].keys()).intersection(
                place_id_clusters[i]) for i in range(num_clusters)]

            max_cluster = np.argmax(num_houses_per_cluster)

            block_id_clusters[max_cluster].add(block)

        tightnesses = [self.calculate_tightness(block_id_clusters[i], place_id_clusters[i]) for i in range(num_clusters)]

        clusters = []
        for i in range(num_clusters):
            clusters.append({
                "block_ids": block_id_clusters[i],
                "place_ids": place_id_clusters[i],
                "tightness": tightnesses[i]
            })

        return clusters

    def __call__(self) -> list[dict[str, set[str] | float]]:
        """
        Cluster a large area into smaller areas suitable for turf split optimization.

        Returns
        -------
            list[dict[str, set[str] | float]]: The clusters of blocks and places, and attributes.
                "block_ids" -> set of block ids in the cluster
                "place_ids" -> set of place ids in the cluster
                "tightness" -> the tightness of the cluster
        """
        return self.cluster_blockwise()
