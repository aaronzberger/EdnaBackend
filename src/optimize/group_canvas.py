import sys
from src.config import BLOCK_DB_IDX, NodeType, Point
from src.distances.mix import MixDistances
from src.optimize.optimizer import Optimizer
from src.optimize.base_solver import ProblemInfo
from src.distances.blocks import BlockDistances
from src.distances.houses import HouseDistances
from src.distances.nodes import NodeDistances
from src.config import MAX_TOURING_DISTANCE
from src.utils.db import Database
from src.utils.gps import great_circle_distance
from src.utils.viz import display_custom_area


class GroupCanvas(Optimizer):
    def __init__(self, block_ids: set[str], place_ids: set[str], voter_ids: set[str], depot: Point, num_routes: int):
        """
        Create a group canvas problem.

        Parameters
        ----------
        block_ids : set[str]
            The blocks to visit.
        place_ids : set[str]
            The places to visit.
        voter_ids : set[str]
            The voters to visit.
        depot : Point
            The depot to start from.
        num_routes : int
            The number of routes to create.
        """
        super().__init__(block_ids=block_ids, place_ids=place_ids, voter_ids=voter_ids)

        self.db = Database()

        self.depot = depot
        radius = MAX_TOURING_DISTANCE / 2

        self.local_block_ids = set()
        self.local_places: list[Point] = []

        # region Retrieve area subsection
        for block_id in block_ids:
            block = self.db.get_dict(block_id, BLOCK_DB_IDX)
            if block is None:
                print(f"Block {block_id} not found in database")
                sys.exit(1)

            if great_circle_distance(block["nodes"][0], self.depot) < radius or great_circle_distance(block["nodes"][-1], self.depot) < radius:
                self.local_block_ids.add(block_id)

                # Find which places on this block are in the universe
                self.matching_place_ids = set(block["places"].keys()).intersection(self.place_ids)

                # Add the places to the list of places to visit
                self.local_places.extend(
                    Point(lat=block["places"][i]["lat"], lon=block["places"][i]["lon"], id=i, type=NodeType.house) for i in self.matching_place_ids)

        display_custom_area(depot=self.depot, places=self.local_places)
        # endregion

        print(f'Of {len(block_ids)} blocks, {len(self.local_block_ids)} are within {radius} meters of the depot')

        # region Load distance matrices
        self.node_distances = NodeDistances(
            block_ids=self.local_block_ids, skip_update=True)

        self.house_distances = HouseDistances(
            block_ids=self.local_block_ids, node_distances=self.node_distances, depots=[self.depot])

        self.mix_distances = MixDistances(
            house_distances=self.house_distances, node_distances=self.node_distances)
        # endregion

        self.build_problem(houses=self.local_places, depot=self.depot, num_routes=num_routes, mix_distances=self.mix_distances)

    def build_problem(
        self, houses: list[Point], depot: Point, num_routes: int, mix_distances: MixDistances,
    ):
        """
        Create a group canvas problem.

        Parameters
        ----------
        houses : list[Point]
            The houses to visit.
        depot : Point
            The depot to start from.
        num_routes : int
            The number of routes to create.
        mix_distances : MixDistances
            The distance matrix to use.
        """
        super().build_problem(mix_distances=mix_distances)

        self.points = [depot] * num_routes + houses

        self.problem_info = ProblemInfo(
            points=self.points,
            num_vehicles=num_routes,
            num_depots=num_routes,
            num_points=len(self.points),
            starts=[i for i in range(num_routes)],
            ends=[i for i in range(num_routes)],
        )
