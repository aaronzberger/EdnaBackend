import sys
from src.config import BLOCK_DB_IDX, NodeType, InternalPoint
from src.distances.mix import MixDistances
from src.optimize.optimizer import Optimizer
from src.optimize.base_solver import ProblemInfo
from src.distances.houses import AbodeDistances
from src.distances.nodes import NodeDistances
from src.config import MAX_TOURING_DISTANCE
from src.utils.db import Database
from src.utils.gps import great_circle_distance
from src.utils.viz import display_custom_area


class GroupCanvas(Optimizer):
    def __init__(
        self,
        block_ids: set[str],
        abode_ids: set[str],
        depot: InternalPoint,
        num_routes: int,
    ):
        """
        Create a group canvas problem.

        Parameters
        ----------
        block_ids : set[str]
            The blocks to visit.
        abode_ids : set[str]
            The abodes to visit.
        depot : Point
            The depot to start from.
        num_routes : int
            The number of routes to create.
        """
        super().__init__(block_ids=block_ids, abode_ids=abode_ids)

        self.db = Database()

        self.num_routes = num_routes
        self.depot = depot
        radius = MAX_TOURING_DISTANCE / 2

        self.local_block_ids = set()
        self.local_abodes: list[InternalPoint] = []

        # region Retrieve area subsection
        for block_id in block_ids:
            block = self.db.get_dict(block_id, BLOCK_DB_IDX)
            if block is None:
                print(f"Block {block_id} not found in database")
                sys.exit(1)

            if (
                great_circle_distance(block["nodes"][0], self.depot) < radius
                or great_circle_distance(block["nodes"][-1], self.depot) < radius
            ):
                self.local_block_ids.add(block_id)

                # Find which abodes on this block are in the universe
                self.matching_abode_ids = set(block["abodes"].keys()).intersection(
                    self.abode_ids
                )

                # Add the abodes to the list of abodes to visit
                self.local_abodes.extend(
                    InternalPoint(
                        lat=block["abodes"][i]["lat"],
                        lon=block["abodes"][i]["lon"],
                        id=i,
                        type=NodeType.abode,
                    )
                    for i in self.matching_abode_ids
                )

        display_custom_area(depot=self.depot, abodes=self.local_abodes)
        # endregion

        # region Load distance matrices
        self.node_distances = NodeDistances(
            block_ids=self.local_block_ids, skip_update=True
        )

        self.abode_distances = AbodeDistances(
            block_ids=self.local_block_ids,
            node_distances=self.node_distances,
            depots=[self.depot],
        )

        self.mix_distances = MixDistances(
            abode_distances=self.abode_distances, node_distances=self.node_distances
        )
        # endregion

        self.build_problem(abode_points=self.local_abodes)

    def build_problem(self, abode_points: list[InternalPoint]):
        """
        Create a group canvas problem.

        Parameters
        ----------
        abode_points : list[InternalPoint]
            The abodes to visit.
        """
        super().build_problem(mix_distances=self.mix_distances)

        self.points = [self.depot] * self.num_routes + abode_points

        self.problem_info = ProblemInfo(
            points=self.points,
            num_vehicles=self.num_routes,
            num_depots=self.num_routes,
            num_points=len(self.points),
            starts=[i for i in range(self.num_routes)],
            ends=[i for i in range(self.num_routes)],
        )
