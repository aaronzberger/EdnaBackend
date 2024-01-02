import itertools
import json
import math
import os
import sys
from copy import deepcopy

from termcolor import colored

from src.config import (
    CAMPAIGN_ID,
    default_distances_path,
    PROBLEM_TYPE,
    Person,
    PlaceGeography,
    PlaceSemantics,
    Problem_Types,
    VIZ_PATH,
    HouseOutput,
    NodeType,
    Point,
    Solution,
    Tour,
    generate_pt_id,
    pt_id,
    details_file,
    PLACE_DB_IDX,
    BLOCK_DB_IDX,
    VOTER_DB_IDX,
)
from src.distances.mix import MixDistances
from src.utils.gps import SubBlock, project_to_line
from src.utils.route import RouteMaker
from src.utils.viz import (
    display_house_orders,
    display_individual_walk_lists,
)
from src.utils.db import Database


class PostProcess:
    def __init__(self, optimizer_points: list[Point], place_ids: set[str], mix_distances: MixDistances):
        # self._all_blocks: blocks_file_t = json.load(open(blocks_file))
        # self.house_id_to_block_id: dict[str, str] = json.load(
        #     open(house_id_to_block_id_file)
        # )
        # self.house_to_voters = json.load(open(house_to_voters_file))

        self.universe_place_ids = place_ids

        self.db = Database()

        # self.requested_blocks = requested_blocks
        self.tour_points = optimizer_points
        RouteMaker()

        # Map block IDs to the UUIDs on this route on that block
        self.blocks_on_route: dict[str, list[str]] = {}
        self.inserted_houses: set[str] = set()

        self.mix_distances = mix_distances

    def _calculate_exit(self, final_house: Point, next_house: Point) -> Point:
        """
        Calculate the optimal exit point for a subsegment given the final house and the next house.

        Parameters
        ----------
            final_house (Point): the final house on the segment
            next_house (Point): the next house to visit after this segment

        Returns
        -------
            Point: the exit point of this segment, which is either the start or endpoint of final_house's segment
        """
        # Determine the exit direction, which will either be the start or end of the segment
        origin_block_id = self.db.get_dict(final_house["id"], PLACE_DB_IDX)["block_id"]
        origin_block = self.db.get_dict(origin_block_id, BLOCK_DB_IDX)

        end_node = deepcopy(origin_block["nodes"][-1])
        end_node["type"] = NodeType.node

        through_end = self.mix_distances.get_distance(p1=end_node, p2=next_house)

        if through_end is not None:
            through_end += origin_block["places"][final_house["id"]]["distance_to_end"]

        start_node = deepcopy(origin_block["nodes"][0])
        start_node["type"] = NodeType.node

        through_start = self.mix_distances.get_distance(p1=start_node, p2=next_house)

        if through_start is not None:
            through_start += origin_block["places"][final_house["id"]][
                "distance_to_start"
            ]

        if through_start is None and through_end is None:
            print(
                colored(
                    f"Unable to find distance through start/end of block (final house {final_house}, next house {next_house}). Quitting.",
                    "red",
                )
            )
            sys.exit(1)

        if through_start is None:
            return origin_block["nodes"][-1]
        if through_end is None:
            return origin_block["nodes"][0]

        return (
            origin_block["nodes"][-1]
            if through_end < through_start
            else origin_block["nodes"][0]
        )

    def _calculate_entrance(self, intersection: Point, next_house: Point) -> Point:
        """
        Calculate the optimal entrance point for a sub-block given the running intersection and the next house.

        Parameters
        ----------
            intersection (Point): the current location of the walker
            next_house (Point): the first house to visit on the next segment

        Returns
        -------
            Point: the entrance point of the next segment, which is either the start or endpoint of next_house's segment
        """
        # Determine the exit direction, which will either be the start or end of the segment
        destination_block_id = self.db.get_dict(next_house["id"], PLACE_DB_IDX)[
            "block_id"
        ]
        destination_block = self.db.get_dict(destination_block_id, BLOCK_DB_IDX)

        intersection["type"] = NodeType.node
        end_node = deepcopy(destination_block["nodes"][-1])
        end_node["type"] = NodeType.node

        through_end = self.mix_distances.get_distance(p1=intersection, p2=end_node)

        if through_end is not None:
            through_end += destination_block["places"][next_house["id"]][
                "distance_to_end"
            ]

        start_node = deepcopy(destination_block["nodes"][0])
        start_node["type"] = NodeType.node

        through_start = self.mix_distances.get_distance(p1=intersection, p2=start_node)

        if through_start is not None:
            through_start += destination_block["places"][next_house["id"]][
                "distance_to_start"
            ]

        if through_start is None and through_end is None:
            print(
                colored(
                    f"Unable to find distance through start/end of block. Final house {intersection}, next house {next_house}. Quitting.",
                    "red",
                )
            )
            sys.exit(1)

        if through_start is None:
            return destination_block["nodes"][-1]
        if through_end is None:
            return destination_block["nodes"][0]

        return (
            destination_block["nodes"][-1]
            if through_end < through_start
            else destination_block["nodes"][0]
        )

    def _split_sub_block(
        self, sub_block: SubBlock, entrance: Point, exit: Point
    ) -> tuple[SubBlock, SubBlock]:
        nav_pts_1 = sub_block.navigation_points[
            : len(sub_block.navigation_points) // 2 + 1
        ]
        nav_pts_2 = sub_block.navigation_points[len(sub_block.navigation_points) // 2 :]

        assert nav_pts_1[-1] == nav_pts_2[0]

        houses_1 = [
            i for i in sub_block.houses if i["subsegment_start"] < len(nav_pts_1) - 1
        ]
        houses_2 = [
            i for i in sub_block.houses if i["subsegment_start"] >= len(nav_pts_1) - 1
        ]

        for i in range(len(houses_2)):
            houses_2[i]["subsegment_start"] -= len(nav_pts_1) - 1

        sub_block_1 = SubBlock(
            navigation_points=nav_pts_1,
            block_id=sub_block.block_id,
            houses=houses_1,
            start=entrance,
            end=nav_pts_1[-1],
            block=sub_block.block,
            extremum=sub_block.extremum,
        )

        sub_block_2 = SubBlock(
            navigation_points=nav_pts_2,
            block_id=sub_block.block_id,
            houses=houses_2,
            start=nav_pts_2[0],
            end=exit,
            block=sub_block.block,
            extremum=sub_block.extremum,
        )

        return sub_block_1, sub_block_2

    def _process_sub_block(
        self, houses: list[Point], block_id: str, entrance: Point, exit: Point
    ) -> SubBlock | tuple[SubBlock, SubBlock]:
        uuids = [i["id"] for i in houses]
        block_houses = {
            uuid: info
            for uuid, info in self.db.get_dict(block_id, BLOCK_DB_IDX)["places"].items()
            if uuid in uuids
        }

        # if len(uuids) != len(self.blocks_on_route[block_id]):
        #     print(
        #         colored(
        #             f"Notice: There are {len(uuids)} houses on this block, but {len(self.blocks_on_route[block_id])} houses on this block in the route" + \
        #             f"\nThe unplaced houses are {set(self.blocks_on_route[block_id]) - set(uuids)}",
        #             "blue",
        #         )
        #     )

        block = self.db.get_dict(block_id, BLOCK_DB_IDX)

        extremum: tuple[Point, Point] = (entrance, exit)

        # region: Calculate the navigation points
        if pt_id(entrance) != pt_id(exit):
            # Order the navigation points (reverse the order if necessary)
            navigation_points = (
                block["nodes"]
                if pt_id(entrance) == pt_id(block["nodes"][0])
                else block["nodes"][::-1]
            )

        elif pt_id(entrance) == pt_id(exit) == pt_id(block["nodes"][0]):
            # User certainly walks at the front of the block, so simply find the
            # last point they need to walk at the back of the block
            extremum_house_uuid, extremum_house = max(
                block_houses.items(), key=lambda a: a[1]["distance_to_start"]
            )
            end_extremum = project_to_line(
                p1=Point(
                    lat=extremum_house["lat"],
                    lon=extremum_house["lon"],
                    type=NodeType.house,
                    id=extremum_house_uuid,
                ),
                p2=block["nodes"][extremum_house["subsegment"][0]],
                p3=block["nodes"][extremum_house["subsegment"][1]],
            )
            extremum = (extremum[0], end_extremum)

            navigation_points = (
                block["nodes"][: extremum_house["subsegment"][0] + 1]
                + [end_extremum]
                + list(reversed(block["nodes"][: extremum_house["subsegment"][0] + 1]))
            )

        elif pt_id(entrance) == pt_id(exit) == pt_id(block["nodes"][-1]):
            extremum_house_uuid, extremum_house = max(
                block_houses.items(), key=lambda a: a[1]["distance_to_end"]
            )
            start_extremum = project_to_line(
                p1=Point(
                    lat=extremum_house["lat"],
                    lon=extremum_house["lon"],
                    type=NodeType.house,
                    id=extremum_house_uuid,
                ),
                p2=block["nodes"][extremum_house["subsegment"][0]],
                p3=block["nodes"][extremum_house["subsegment"][1]],
            )
            extremum = (start_extremum, extremum[1])

            navigation_points = (
                list(reversed(block["nodes"][extremum_house["subsegment"][1] :]))
                + [start_extremum]
                + block["nodes"][extremum_house["subsegment"][1] :]
            )

        else:
            print(
                colored(
                    'Error: "entrance" and "exit" were not valid for this block', "red"
                )
            )
        # endregion

        unused_uuids = (
            set(self.blocks_on_route[block_id]) - set(uuids) - self.inserted_houses
        )

        # region: Order the houses
        if pt_id(entrance) != pt_id(exit):
            running_side = block_houses[houses[0]["id"]]["side"]
            start = 0
            i = 0
            new_house_order = []
            running_distance_to_start = 0
            added_houses = set()

            metric = (
                "distance_to_start"
                if pt_id(entrance) == pt_id(block["nodes"][0])
                else "distance_to_end"
            )

            for i, house in enumerate(houses):
                if block_houses[house["id"]]["side"] != running_side:
                    # Add any houses on the same side with less distance to the start
                    new_houses = deepcopy(houses[start:i])
                    new_houses = [h for h in new_houses if h["id"] not in added_houses]

                    running_distance_to_start = max(
                        running_distance_to_start,
                        max(
                            [
                                block_houses[house["id"]][metric]
                                for house in new_houses + new_house_order
                            ]
                        ),
                    )

                    for remaining_house in houses[i + 1 :]:
                        if (
                            block_houses[remaining_house["id"]]["side"] == running_side
                            and block_houses[remaining_house["id"]][metric]
                            < running_distance_to_start
                            and remaining_house["id"] not in added_houses
                        ):
                            new_houses.append(remaining_house)

                    # Also, add any houses on future blocks which can be moved up to this block
                    for potential_house_id in unused_uuids:
                        info: PlaceGeography = self.db.get_dict(block_id, BLOCK_DB_IDX)[
                            "places"
                        ][potential_house_id]
                        if (
                            info["side"] != running_side
                            and info[metric] < running_distance_to_start
                            and potential_house_id not in added_houses
                        ):
                            new_houses.append(
                                Point(
                                    lat=info["lat"],
                                    lon=info["lon"],
                                    type=NodeType.house,
                                    id=potential_house_id,
                                )
                            )

                            uuids.append(potential_house_id)
                            block_houses[potential_house_id] = info

                            # print(f"Moving up house with id {potential_house_id} to this block (different entrance/exit)")

                    new_house_order += sorted(
                        new_houses,
                        key=lambda h, metric=metric: block_houses[h["id"]][metric],
                    )
                    running_side = block_houses[house["id"]]["side"]
                    start = i
                    added_houses.update([h["id"] for h in new_houses])

            # Now, sort the last side
            houses_left = [h for h in houses[start:] if h["id"] not in added_houses]

            # Also, add any houses on future blocks which can be moved up to this block
            for potential_house_id in unused_uuids:
                info: PlaceGeography = self.db.get_dict(block_id, BLOCK_DB_IDX)[
                    "places"
                ][potential_house_id]
                # info = self.requested_blocks[block_id]["addresses"][potential_house_id]
                if (
                    info["side"] == running_side
                    and potential_house_id not in added_houses
                ):
                    houses_left.append(
                        Point(
                            lat=info["lat"],
                            lon=info["lon"],
                            type=NodeType.house,
                            id=potential_house_id,
                        )
                    )

                    uuids.append(potential_house_id)
                    block_houses[potential_house_id] = info

                    # print(f"Moving up house with id {potential_house_id} to this block (different entrance/exit)")

            new_house_order += sorted(
                houses_left, key=lambda h: block_houses[h["id"]][metric]
            )

            houses = new_house_order

            # We're always going forward, so the subsegments are as they are
            for i, house in enumerate(houses):
                sub_start = block["nodes"][block_houses[house["id"]]["subsegment"][0]]
                sub_end = block["nodes"][block_houses[house["id"]]["subsegment"][1]]
                sub_start_idx = navigation_points.index(sub_start)
                sub_end_idx = navigation_points.index(sub_end)
                houses[i]["subsegment_start"] = min(sub_start_idx, sub_end_idx)

        elif pt_id(entrance) == pt_id(exit):
            # The optimal path is always to go out on one side and back on the other
            # (since you must go out and back anyway, and this minimizes street crossings)
            out_side = [
                h
                for h in houses
                if block_houses[h["id"]]["side"]
                == block_houses[houses[0]["id"]]["side"]
            ]
            back_side = [
                h
                for h in houses
                if block_houses[h["id"]]["side"]
                != block_houses[houses[0]["id"]]["side"]
            ]

            # To avoid mis-placing houses off the end, sort by the distance to the entrance/exit
            metric_out = (
                "distance_to_start"
                if pt_id(entrance) == pt_id(block["nodes"][0])
                else "distance_to_end"
            )

            # Put the "out" side houses first, then the "back" side houses
            houses = sorted(
                out_side,
                key=lambda h: block_houses[h["id"]][metric_out],
            ) + sorted(
                back_side, key=lambda h: block_houses[h["id"]][metric_out], reverse=True
            )

            # For the out houses, we're always going forward, so the subsegments are as they are
            for i, house in enumerate(out_side):
                out_nav_nodes = navigation_points[: len(navigation_points) // 2 + 1]
                # print('OUT NAV', out_nav_nodes, flush=True)
                sub_start = block["nodes"][block_houses[house["id"]]["subsegment"][0]]
                sub_end = block["nodes"][block_houses[house["id"]]["subsegment"][1]]

                # print(f'Out nav nodes are {out_nav_nodes} for house {house["id"]}, subsegment {sub_start}, {sub_end}')

                sub_start_idx = sub_end_idx = len(out_nav_nodes)
                for node in out_nav_nodes:
                    if generate_pt_id(node) == generate_pt_id(sub_start):
                        sub_start_idx = out_nav_nodes.index(node)
                    if generate_pt_id(node) == generate_pt_id(sub_end):
                        sub_end_idx = out_nav_nodes.index(node)

                if min(sub_start_idx, sub_end_idx) == len(out_nav_nodes):
                    sub_start_idx = len(out_nav_nodes) - 1
                    print(
                        colored(
                            f"WARNNING: house {house['id']} 's subsegment is no longer existent: extremum calculation likely failed",
                            "yellow",
                        )
                    )

                houses[i]["subsegment_start"] = min(sub_start_idx, sub_end_idx)

            # For the back houses, they are on the second half of the subsegments
            for i, house in enumerate(back_side):
                back_nav_nodes = navigation_points[len(navigation_points) // 2 :]
                # print('BACK NAV', back_nav_nodes, flush=True)
                sub_start = block["nodes"][block_houses[house["id"]]["subsegment"][0]]
                sub_end = block["nodes"][block_houses[house["id"]]["subsegment"][1]]

                # print(f'Back nav nodes are {back_nav_nodes} for house {house["id"]}, subsegment {sub_start}, {sub_end}')

                sub_start_idx = sub_end_idx = len(back_nav_nodes)
                for node in back_nav_nodes:
                    if generate_pt_id(node) == generate_pt_id(sub_start):
                        sub_start_idx = back_nav_nodes.index(node)
                    if generate_pt_id(node) == generate_pt_id(sub_end):
                        sub_end_idx = back_nav_nodes.index(node)

                if min(sub_start_idx, sub_end_idx) == len(back_nav_nodes):
                    sub_start_idx = len(back_nav_nodes) - 1
                    print(
                        colored(
                            f"WARNNING: house {house['id']} 's subsegment is no longer existent: extremum calculation likely failed",
                            "yellow",
                        )
                    )

                houses[i + len(out_side)]["subsegment_start"] = (
                    min(sub_start_idx, sub_end_idx) + len(navigation_points) // 2 - 1
                )

        # endregion

        sub_block = SubBlock(
            block=block,
            block_id=block_id,
            start=entrance,
            end=exit,
            extremum=extremum,
            houses=houses,
            navigation_points=navigation_points,
        )

        if pt_id(entrance) == pt_id(exit):
            new_dual = self._split_sub_block(sub_block, entrance, exit)
            return new_dual

        for uuid in uuids:
            self.inserted_houses.add(uuid)

        return sub_block

    def fill_holes(self, walk_list: list[SubBlock]) -> list[SubBlock]:
        """
        Fill in intermediate blocks between sub-blocks, wherever the end of one block
        is not the same as the start of the next one.

        Parameters
        ----------
            walk_list (list[SubBlock]): The list of sub-blocks to fill in

        Returns
        -------
            list[SubBlock]: The list of sub-blocks with holes filled in
        """
        # Iterate through the solution and add subsegments
        new_walk_list: list[SubBlock] = []

        ids = []
        for subblock in walk_list:
            ids.append(pt_id(subblock.start))
            ids.append(pt_id(subblock.end))

        for first, second in itertools.pairwise(walk_list):
            new_walk_list.append(first)

            # If the end of the first subblock is not the same as the start of the second subblock, add a new subblock
            if pt_id(first.end) != pt_id(second.start):
                route_info = RouteMaker.get_route(first.end, second.start)

                if route_info is None:
                    print(
                        colored(
                            "Not adding route between sub-blocks because no route was found",
                        ),
                        "yellow",
                    )
                    continue
                block_ids, distance = route_info

                running_node = first.end
                for block_id in block_ids:
                    # TODO: Check if we can remove this. It is the only reference to all_blocks
                    # May be able to just refer to block_id in the subblock
                    block = self.db.get_dict(block_id, BLOCK_DB_IDX)

                    start = Point(
                        lat=block["nodes"][0]["lat"],
                        lon=block["nodes"][0]["lon"],
                        type=NodeType.node,
                        id=generate_pt_id(
                            block["nodes"][0]["lat"], block["nodes"][0]["lon"]
                        ),
                    )
                    end = Point(
                        lat=block["nodes"][-1]["lat"],
                        lon=block["nodes"][-1]["lon"],
                        type=NodeType.node,
                        id=generate_pt_id(
                            block["nodes"][-1]["lat"], block["nodes"][-1]["lon"]
                        ),
                    )

                    reverse = pt_id(start) != pt_id(running_node)
                    nav_pts = block["nodes"]
                    if reverse:
                        start, end = end, start
                        nav_pts = nav_pts[::-1]

                    assert pt_id(start) == pt_id(
                        running_node
                    ), f"{start} != {running_node}"
                    assert pt_id(nav_pts[0]) == pt_id(start), f"{nav_pts[0]} != {start}"

                    running_node = end

                    new_walk_list.append(
                        SubBlock(
                            block=block,
                            block_id=block_id,
                            start=start,
                            end=end,
                            extremum=(start, end),
                            houses=[],
                            navigation_points=nav_pts,
                        )
                    )

        new_walk_list.append(walk_list[-1])

        return new_walk_list

    def post_process(self, tour: Tour) -> list[SubBlock]:
        # Iterate through the solution and add subsegments
        walk_list: list[SubBlock] = []

        # From the index of each stop, get the points for those stops
        tour_stops: list[Point] = [
            self.tour_points[h["location"]["index"]] for h in tour["stops"]
        ]
        depot, houses = tour_stops[0], tour_stops[1:-1]
        self.depot = depot

        for house in houses:
            if (
                self.db.get_dict(house["id"], PLACE_DB_IDX)["block_id"]
                not in self.blocks_on_route
            ):
                self.blocks_on_route[
                    self.db.get_dict(house["id"], PLACE_DB_IDX)["block_id"]
                ] = [house["id"]]
            else:
                self.blocks_on_route[
                    self.db.get_dict(house["id"], PLACE_DB_IDX)["block_id"]
                ].append(house["id"])

        current_sub_block_houses: list[Point] = []

        # Take the side closest to the first house (likely where a canvasser would park)
        running_intersection = depot
        running_block_id = self.db.get_dict(houses[0]["id"], PLACE_DB_IDX)["block_id"]

        # Process the list
        for house, next_house in itertools.pairwise(houses):
            next_block_id = self.db.get_dict(next_house["id"], PLACE_DB_IDX)["block_id"]

            if house["id"] not in self.inserted_houses:
                current_sub_block_houses.append(house)

            if next_block_id != running_block_id:
                if len(current_sub_block_houses) == 0:
                    running_block_id = next_block_id
                    continue

                # Calculate the entrance to the block which is ending
                entrance_pt = self._calculate_entrance(
                    running_intersection, current_sub_block_houses[0]
                )
                # Calculate the exit from the block which is ending
                exit_pt = self._calculate_exit(house, next_house)
                sub_block = self._process_sub_block(
                    current_sub_block_houses,
                    running_block_id,
                    entrance=entrance_pt,
                    exit=exit_pt,
                )

                if isinstance(sub_block, SubBlock):
                    walk_list.append(sub_block)
                else:
                    walk_list.extend(sub_block)

                current_sub_block_houses = []

                # After completing this segment, the canvasser is at the end of the subsegment
                running_intersection = exit_pt
                running_block_id = next_block_id

        # Since we used pairwise, the last house is never evaluated
        if houses[-1]["id"] not in self.inserted_houses:
            current_sub_block_houses.append(houses[-1])

        if len(current_sub_block_houses) != 0:
            # Determine the final intersection the canvasser will end up at to process the final subsegment
            exit_point = self._calculate_entrance(depot, current_sub_block_houses[-1])
            entrance_point = self._calculate_entrance(
                running_intersection, current_sub_block_houses[0]
            )

            sub_block = self._process_sub_block(
                current_sub_block_houses,
                running_block_id,
                entrance=entrance_point,
                exit=exit_point,
            )

            if isinstance(sub_block, SubBlock):
                walk_list.append(sub_block)
            else:
                walk_list.extend(sub_block)

        # Fill in any holes
        walk_list = self.fill_holes(walk_list)

        # If the final sub-block is empty, remove it
        if len(walk_list[-1].houses) == 0:
            walk_list.pop()

        return walk_list

    def generate_file(
        self, walk_list: list[SubBlock], output_file: str, id: str, form: dict
    ):
        """
        Generate a JSON file with the walk list (for front-end).

        Parameters
        ----------
            walk_list (list[SubBlock]): The walk list to generate the file for
            output_file (str): The file to write to
        """
        # Generate the JSON file
        list_out = {}
        list_out["id"] = id
        list_out["blocks"] = []
        for sub_block in walk_list:
            nodes = []
            for nav_pt in sub_block.navigation_points:
                nodes.append({"lat": nav_pt["lat"], "lon": nav_pt["lon"]})
            houses = []
            for house in sub_block.houses:
                # Lookup this entry by uuid
                try:
                    place: PlaceSemantics = self.db.get_dict(house["id"], PLACE_DB_IDX)

                    if isinstance(place["voters"], list):
                        voters: list[Person] = [
                            self.db.get_dict(v, VOTER_DB_IDX) for v in place["voters"]
                        ]
                    else:
                        # TODO: Deal with apartments
                        voters = []

                    place_geo: PlaceGeography = self.db.get_dict(
                        place["block_id"], BLOCK_DB_IDX
                    )["places"][house["id"]]
                except KeyError:
                    print(
                        colored(
                            "House {} with id {} not found in voter info".format(
                                house, pt_id(house)
                            ),
                            "red",
                        )
                    )
                    sys.exit(1)

                assert (
                    place_geo["lat"] == house["lat"]
                    and place_geo["lon"] == house["lon"]
                ), "House coordinate mismatch: voter info file had {}, routing file had {}".format(
                    (place_geo["lat"], place_geo["lon"]),
                    (house["lat"], house["lon"]),
                )

                houses.append(
                    HouseOutput(
                        display_address=place["display_address"],
                        city=place["city"],
                        state=place["state"],
                        zip=place["zip"],
                        uuid=pt_id(house),
                        latitude=place_geo["lat"],
                        longitude=place_geo["lon"],
                        voter_info=voters,
                        subsegment_start=house["subsegment_start"],
                    )
                )

            list_out["blocks"].append({"nodes": nodes, "houses": houses})

        list_out["form"] = form

        # Write the file
        json.dump(list_out, open(output_file, "w"))


def get_distance_cost(idx1: int, idx2: int, distances_file: str) -> tuple[float, float]:
    """
    Get the distance and cost from the routing input for two indices.

    Parameters
    ----------
        idx1 (int): The first index
        idx2 (int): The second index

    Returns
    -------
        float: The distance between the two indices
        float: The cost between the two indices
    """
    distance_costs = json.load(open(distances_file))

    # Distances are stored as a flattened matrix
    # The index of the distance between two points is calculated as follows:
    #   (idx1 * num_points) + idx2
    #   (idx2 * num_points) + idx1

    num_points = math.sqrt(len(distance_costs["distances"]))
    assert num_points == int(num_points), "Number of points is not an integer"

    return (
        distance_costs["travelTimes"][(idx1 * int(num_points)) + idx2],
        distance_costs["distances"][(idx1 * int(num_points)) + idx2],
    )


def process_solution(
    solution: Solution,
    optimizer_points: list[Point],
    place_ids: set[str],
    mix_distances: MixDistances,
    viz_path: str = VIZ_PATH,
    distances_path: str = default_distances_path,
    id: str = CAMPAIGN_ID,
):
    point_orders: list[list[tuple[Point, int]]] = []

    for i, route in enumerate(solution["tours"]):
        point_orders.append([])
        for stop in route["stops"][1:-1]:
            point_orders[i].append(
                (optimizer_points[stop["location"]["index"]], stop["location"]["index"])
            )

    # Ensure that every place in point orders is in the universe
    # for sublist in point_orders:
    #     for pt in sublist:
    #         assert pt[0]["id"] in place_ids, f"Point {pt[0]['id']} not in universe"


    # display_distance_matrix(
    #     optimizer_points, os.path.join(problem_path, "distances.json")
    # ).save(os.path.join(viz_path, "distances.html"))

    # house_dcs = [[HouseDistances.get_distance(i, j) for (i, j) in itertools.pairwise(list)] for list in point_orders]
    house_dcs = [
        [
            get_distance_cost(i[1], j[1], default_distances_path)
            for (i, j) in itertools.pairwise(list)
        ]
        for list in point_orders
    ]

    points = [[i[0] for i in list] for list in point_orders]

    display_house_orders(points, dcs=house_dcs).save(
        os.path.join(viz_path, "direct_output.html")
    )

    if os.path.exists(details_file):
        details = json.load(open(details_file))
    else:
        details = {}

    post_processor = PostProcess(optimizer_points=optimizer_points, place_ids=place_ids, mix_distances=mix_distances)
    walk_lists: list[list[SubBlock]] = []
    for i, tour in enumerate(solution["tours"]):
        # Do not count the startingg location service at the start or end
        tour["stops"] = (
            tour["stops"][1:-1]
            if PROBLEM_TYPE == Problem_Types.turf_split
            else tour["stops"]
        )

        if len(tour["stops"]) == 0:
            print(f"List {i} has 0 stops")
            continue

        walk_lists.append(post_processor.post_process(tour))

        list_id = f"{CAMPAIGN_ID}-{id}-{i}"
        if list_id in details:
            print(colored(f"Warning: List {list_id} already exists in details"))

        num_houses = sum([len(sub_block.houses) for sub_block in walk_lists[-1]])
        distance = 0
        for sub_block in walk_lists[-1]:
            distance += sub_block.length
        start_point = {
            "lat": walk_lists[-1][0].start["lat"],
            "lon": walk_lists[-1][0].start["lon"],
        }

        details[list_id] = {
            "distance": distance,
            "num_houses": num_houses,
            "start_point": start_point,
        }

    json.dump(details, open(details_file, "w"))

    all_list_places = set()
    for walk_list in walk_lists:
        for sub_block in walk_list:
            for house in sub_block.houses:
                all_list_places.add(house["id"])

    for id in all_list_places:
        assert id in place_ids, f"Place {id} not in universe"

    list_visualizations = display_individual_walk_lists(walk_lists)
    for i, walk_list in enumerate(list_visualizations):
        walk_list.save(os.path.join(viz_path, f"{CAMPAIGN_ID}-{id}-{i}.html"))

    form = json.load(
        open(os.path.join("regions", CAMPAIGN_ID, "input", "form.json"), "r")
    )

    for i in range(len(walk_lists)):
        list_id = f"{CAMPAIGN_ID}-{id}-{i}"
        post_processor.generate_file(
            walk_lists[i],
            os.path.join(viz_path, f"{list_id}.json"),
            id=list_id,
            form=form,
        )


# if __name__ == "__main__":
#     # Load the requested blocks
#     requested_blocks = json.load(open(requested_blocks_file, "r"))

#     # Generate node distance matrix
#     NodeDistances(requested_blocks)

#     # Generate block distance matrix
#     BlockDistances(requested_blocks)

#     # Initialize calculator for mixed distances
#     MixDistances()

#     if len(sys.argv) == 2:
#         problem_dir = os.path.join(
#             BASE_DIR, "regions", CAMPAIGN_NAME, "areas", sys.argv[1], "problem"
#         )
#         viz_dir = os.path.join(
#             BASE_DIR, "regions", CAMPAIGN_NAME, "areas", sys.argv[1], "viz"
#         )
#         solution = Optimizer.process_solution(
#             os.path.join(problem_dir, "solution.json")
#         )

#         optimizer_points = pickle.load(
#             open(os.path.join(problem_dir, "points.pkl"), "rb")
#         )

#         process_solution(
#             solution=solution,
#             optimizer_points=optimizer_points,
#             requested_blocks=requested_blocks,
#             viz_path=viz_dir,
#             problem_path=problem_dir,
#             id=sys.argv[1],
#         )
