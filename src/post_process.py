import itertools
import json
import math
import os
import pickle
from copy import deepcopy
from random import randint
import sys

import names
from termcolor import colored

from src.config import (
    BASE_DIR,
    NODE_TOO_FAR_DISTANCE,
    PROBLEM_PATH,
    TURF_SPLIT,
    VIZ_PATH,
    HouseOutput,
    HousePeople,
    NodeType,
    Person,
    Point,
    Solution,
    Tour,
    blocks_file,
    blocks_file_t,
    generate_pt_id,
    house_id_to_block_id_file,
    optimizer_points_pickle_file,
    pt_id,
    requested_blocks_file,
    house_to_voters_file
)
from src.distances.blocks import BlockDistances
from src.distances.houses import HouseDistances
from src.distances.mix import MixDistances
from src.distances.nodes import NodeDistances
from src.gps_utils import SubBlock, project_to_line
from src.optimize import Optimizer
from src.route import RouteMaker
from src.viz_utils import (
    display_distance_matrix,
    display_house_orders,
    display_individual_walk_lists,
    display_walk_lists,
)


class PostProcess:
    def __init__(self, blocks: blocks_file_t, points: list[Point]):
        self._all_blocks: blocks_file_t = json.load(open(blocks_file))
        self.house_id_to_block_id: dict[str, str] = json.load(
            open(house_id_to_block_id_file)
        )
        self.house_to_voters = json.load(open(house_to_voters_file))
        # self.address_to_segment_id: houses_file_t = json.load(open(addresses_file))

        self.blocks = blocks
        self.points = points
        RouteMaker()

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
        origin_block_id = self.house_id_to_block_id[final_house["id"]]
        origin_block = self.blocks[origin_block_id]

        end_node = deepcopy(origin_block["nodes"][-1])
        end_node["type"] = NodeType.node

        assert next_house["type"] == NodeType.house

        through_end = MixDistances.get_distance(p1=end_node, p2=next_house)

        if through_end is not None:
            through_end = self.blocks[origin_block_id]["addresses"][final_house["id"]][
                "distance_to_end"
            ]

        start_node = deepcopy(origin_block["nodes"][0])
        start_node["type"] = NodeType.node

        through_start = MixDistances.get_distance(p1=start_node, p2=next_house)

        if through_start is not None:
            through_start = self.blocks[origin_block_id]["addresses"][final_house["id"]][
                "distance_to_start"
            ]

        if through_start is None and through_end is None:
            print(
                colored(
                    "Unable to find distance through start or end of block in post-processing. Quitting.",
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
        destination_block_id = self.house_id_to_block_id[next_house["id"]]
        destination_block = self.blocks[destination_block_id]

        through_end = self.blocks[destination_block_id]["addresses"][next_house["id"]][
            "distance_to_end"
        ]
        try:
            to_end = NodeDistances.get_distance(
                intersection, destination_block["nodes"][-1]
            )
            through_end += 1600 if to_end is None else to_end
        except TypeError:
            print("Unable to find distance through end of block in post-processing")
            through_end += 1600

        through_start = self.blocks[destination_block_id]["addresses"][
            next_house["id"]
        ]["distance_to_start"]
        try:
            to_start = NodeDistances.get_distance(
                intersection, destination_block["nodes"][0]
            )
            through_start += 1600 if to_start is None else to_start
        except TypeError:
            print("Unable to find distance through end of block in post-processing")
            through_start += 1600

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
        assigned_addresses = [i["id"] for i in houses]
        block_addresses = {
            add: inf
            for add, inf in self.blocks[block_id]["addresses"].items()
            if add in assigned_addresses
        }
        block = self.blocks[block_id]

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
            extremum_house_uuid, extremum_house = max(block_addresses.items(), key=lambda a: a[1]["distance_to_start"])
            end_extremum = project_to_line(
                p1=Point(lat=extremum_house["lat"], lon=extremum_house["lon"], type=NodeType.house, id=extremum_house_uuid),
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
            extremum_house_uuid, extremum_house = max(block_addresses.items(), key=lambda a: a[1]["distance_to_end"])
            start_extremum = project_to_line(
                p1=Point(lat=extremum_house["lat"], lon=extremum_house["lon"], type=NodeType.house, id=extremum_house_uuid),
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

        # region: Order the houses
        if pt_id(entrance) != pt_id(exit):
            running_side = block_addresses[houses[0]["id"]]["side"]
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
                if block_addresses[house["id"]]["side"] != running_side:
                    # Add any houses on the same side with less distance to the start
                    new_houses = deepcopy(houses[start:i])
                    new_houses = [h for h in new_houses if h["id"] not in added_houses]

                    running_distance_to_start = max(
                        running_distance_to_start,
                        max(
                            [
                                block_addresses[house["id"]][metric]
                                for house in new_houses + new_house_order
                            ]
                        ),
                    )

                    for remaining_house in houses[i + 1 :]:
                        if (
                            block_addresses[remaining_house["id"]]["side"]
                            != running_side
                            and block_addresses[remaining_house["id"]][metric]
                            < running_distance_to_start
                            and remaining_house["id"] not in added_houses
                        ):
                            new_houses.append(remaining_house)

                            print(
                                "Adding house {} with {}".format(
                                    remaining_house["id"], house["id"]
                                )
                            )

                    new_house_order += sorted(
                        new_houses,
                        key=lambda h, metric=metric: block_addresses[h["id"]][metric],
                    )
                    running_side = block_addresses[house["id"]]["side"]
                    start = i
                    added_houses.update([h["id"] for h in new_houses])

            # Now, sort the last side
            houses_left = [h for h in houses[start:] if h["id"] not in added_houses]
            new_house_order += sorted(
                houses_left, key=lambda h: block_addresses[h["id"]][metric]
            )

            houses = new_house_order

            # We're always going forward, so the subsegments are as they are
            for i, house in enumerate(houses):
                sub_start = block["nodes"][
                    block_addresses[house["id"]]["subsegment"][0]
                ]
                sub_end = block["nodes"][block_addresses[house["id"]]["subsegment"][1]]
                sub_start_idx = navigation_points.index(sub_start)
                sub_end_idx = navigation_points.index(sub_end)
                houses[i]["subsegment_start"] = min(sub_start_idx, sub_end_idx)

        elif pt_id(entrance) == pt_id(exit):
            # We can assume that the first house is on the "out" side
            # TODO/NOTE: Eventually, we should make the out side be on the same side as where we're coming from (avoid crossing)
            out_side = [
                h
                for h in houses
                if block_addresses[h["id"]]["side"]
                == block_addresses[houses[0]["id"]]["side"]
            ]
            back_side = [
                h
                for h in houses
                if block_addresses[h["id"]]["side"]
                != block_addresses[houses[0]["id"]]["side"]
            ]

            # Put the "out" side houses first, then the "back" side houses
            houses = sorted(
                out_side,
                key=lambda h: block_addresses[h["id"]]["distance_to_start"],
                reverse=generate_pt_id(entrance) != generate_pt_id(block["nodes"][0]),
            ) + sorted(
                back_side,
                key=lambda h: block_addresses[h["id"]]["distance_to_end"],
                reverse=generate_pt_id(entrance) != generate_pt_id(block["nodes"][0]),
            )

            # For the out houses, we're always going forward, so the subsegments are as they are
            # print('ALL NAV', navigation_points, flush=True)
            for i, house in enumerate(out_side):
                out_nav_nodes = navigation_points[: len(navigation_points) // 2 + 1]
                # print('OUT NAV', out_nav_nodes, flush=True)
                sub_start = block["nodes"][
                    block_addresses[house["id"]]["subsegment"][0]
                ]
                sub_end = block["nodes"][block_addresses[house["id"]]["subsegment"][1]]
                try:
                    sub_start_idx = out_nav_nodes.index(sub_start)
                except ValueError:
                    sub_start_idx = len(out_nav_nodes)
                try:
                    sub_end_idx = out_nav_nodes.index(sub_end)
                except ValueError:
                    sub_end_idx = len(out_nav_nodes)

                assert min(sub_start_idx, sub_end_idx) != len(
                    out_nav_nodes
                ), f'House {house["id"]} not found in navigation points'
                houses[i]["subsegment_start"] = min(sub_start_idx, sub_end_idx)

            # For the back houses, they are on the second half of the subsegments
            for i, house in enumerate(back_side):
                back_nav_nodes = navigation_points[len(navigation_points) // 2 :]
                # print('BACK NAV', back_nav_nodes, flush=True)
                sub_start = block["nodes"][
                    block_addresses[house["id"]]["subsegment"][0]
                ]
                sub_end = block["nodes"][block_addresses[house["id"]]["subsegment"][1]]
                try:
                    sub_start_idx = back_nav_nodes.index(sub_start)
                except ValueError:
                    sub_start_idx = len(back_nav_nodes)
                try:
                    sub_end_idx = back_nav_nodes.index(sub_end)
                except ValueError:
                    sub_end_idx = len(back_nav_nodes)

                assert min(sub_start_idx, sub_end_idx) != len(
                    back_nav_nodes
                ), f'House {house["id"]} is not on the back side of the block'
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

        # if pt_id(entrance) == pt_id(exit):
        #     new_dual = self._split_sub_block(sub_block, entrance, exit)
        #     return new_dual

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
                block_ids, distance = RouteMaker.get_route(first.end, second.start)

                running_node = first.end
                for block_id in block_ids:
                    # TODO: Check if we can remove this. It is the only reference to all_blocks
                    # May be able to just refer to block_id in the subblock
                    block = self._all_blocks[block_id]

                    start = Point(
                        lat=block["nodes"][0]["lat"],
                        lon=block["nodes"][0]["lon"],
                        type=NodeType.node,
                        id=generate_pt_id(block["nodes"][0]["lat"], block["nodes"][0]["lon"]),
                    )
                    end = Point(
                        lat=block["nodes"][-1]["lat"],
                        lon=block["nodes"][-1]["lon"],
                        type=NodeType.node,
                        id=generate_pt_id(block["nodes"][-1]["lat"], block["nodes"][-1]["lon"])
                    )

                    reverse = pt_id(start) != pt_id(running_node)
                    nav_pts = block["nodes"]
                    if reverse:
                        start, end = end, start
                        nav_pts = nav_pts[::-1]

                    assert pt_id(start) == pt_id(running_node), f"{start} != {running_node}"
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

    def combine_sub_blocks(self, sub_blocks: list[SubBlock]) -> list[SubBlock]:
        """
        Combine sub-blocks that are on the same block, if there are houses on the same side.
        """
        # new_sub_blocks: list[SubBlock] = []

        for i, sub_block in enumerate(sub_blocks):
            # print("Original navigation points", sub_block.navigation_points, flush=True)
            if len(sub_block.houses) == 0:
                # new_sub_blocks.append(sub_block)
                continue

            new_houses = sub_block.houses.copy()
            block_id = sub_block.block_id
            # block_id = self.address_to_segment_id[sub_block.houses[0]["id"]]

            sub_block_nav_ids = set([pt_id(pt) for pt in sub_block.navigation_points])
            assigned_addresses = [h["id"] for h in sub_block.houses]
            block_addresses = {
                add: inf
                for add, inf in self.blocks[block_id]["addresses"].items()
                if add in assigned_addresses
            }
            house_sides = set([inf["side"] for inf in block_addresses.values()])

            combined_block_addresses = block_addresses.copy()

            # print(block_addresses, flush=True)
            # print(sub_block.houses, flush=True)

            assert len(block_addresses) == len(
                sub_block.houses
            ), "Block addresses was {} but sub block houses was {}".format(
                len(block_addresses), len(sub_block.houses)
            )

            # Check if there are any other sub-blocks for which the navigation points are the same (or reversed)
            for other_sub_block in sub_blocks:
                if sub_block == other_sub_block or len(other_sub_block.houses) == 0:
                    continue

                other_sub_block_nav_ids = set(
                    [pt_id(pt) for pt in other_sub_block.navigation_points]
                )

                # If the navigation points are not the same then skip
                if set(sub_block_nav_ids) != set(other_sub_block_nav_ids):
                    continue

                # print("Combining sub-block {} with sub-block {}".format(0 if len(sub_block.houses) == 0 else sub_block.houses[0],
                #                                                         0 if len(other_sub_block.houses) == 0 else other_sub_block.houses[0]))
                # print("FIrst block has sides {}".format(house_sides))
                # print("Second blok has {} houses".format(len(other_sub_block.houses)))

                other_assigned_addresses = [h["id"] for h in other_sub_block.houses]
                other_block_addresses = {
                    uuid: inf
                    for uuid, inf in self.blocks[block_id]["addresses"].items()
                    if uuid in other_assigned_addresses
                }

                assert len(other_block_addresses) == len(
                    other_sub_block.houses
                ), "Other block addresses was {} but other sub block houses was {}. ".format(
                    len(other_block_addresses), len(other_sub_block.houses)
                )

                # Combine houses on the second block into the first block (if they are on the same side)
                to_delete = []
                # print(len(list(other_block_addresses.values())), len(other_sub_block.houses))
                for info, house in zip(
                    other_block_addresses.values(), other_sub_block.houses
                ):
                    # print(info['side'], house)
                    if info["side"] in house_sides:
                        # print("Adding house {} to sub-block".format(house))
                        new_houses.append(house)
                        to_delete.append(house)

                # Remove the houses from the other sub-block
                for house in to_delete:
                    other_sub_block.houses.remove(house)

                combined_block_addresses = {
                    **combined_block_addresses,
                    **other_block_addresses,
                }

            # Order the houses
            start = 0
            i = 0
            new_house_order = []
            running_distance_to_start = 0
            added_houses = set()

            # combined_block_addresses = {**block_addresses, **other_block_addresses}

            # One of the two endpoints must be an endpoint of the block
            # print(sub_block.start, sub_block.navigation_points[0], sub_block.navigation_points[-1], sub_block.end)
            # assert sub_block.start in [
            #     sub_block.navigation_points[0],
            #     sub_block.navigation_points[-1],
            # ] or sub_block.end in [
            #     sub_block.navigation_points[0],
            #     sub_block.navigation_points[-1],
            # ]

            metric = (
                "distance_to_start"
                if sub_block.navigation_points[0] == self.blocks[block_id]["nodes"][0]
                or sub_block.navigation_points[-1] == self.blocks[block_id]["nodes"][-1]
                else "distance_to_end"
            )

            assert len(new_houses) >= len(sub_block.houses)

            # If it's only one side of the street, just order them
            if len(house_sides) == 1:
                new_house_order = sorted(
                    new_houses, key=lambda h: combined_block_addresses[h["id"]][metric]
                )
            else:
                new_house_order = sorted(
                    new_houses, key=lambda h: combined_block_addresses[h["id"]][metric]
                )
                # running_side = combined_block_addresses[new_houses[0]['id']]['side']
                # for j, house in enumerate(new_houses):
                #     if combined_block_addresses[house['id']]['side'] != running_side:
                #         # Add any houses on the same side with less distance to the start
                #         new_houses = deepcopy(new_houses[start:j])
                #         new_houses = [h for h in new_houses if h['id'] not in added_houses]

                #         running_distance_to_start = max(running_distance_to_start, max([combined_block_addresses[house['id']][metric] for house in new_houses + new_house_order]))

                #         # for remaining_house in new_houses[j + 1:]:
                #         #     if combined_block_addresses[remaining_house['id']]['side'] != running_side and \
                #         #             combined_block_addresses[remaining_house['id']][metric] < running_distance_to_start and \
                #         #             remaining_house['id'] not in added_houses:
                #         #         new_houses.append(remaining_house)

                #                 # print("Adding house {} with {}".format(remaining_house['id'], house['id']))

                #         new_house_order += sorted(new_houses, key=lambda h, metric=metric: combined_block_addresses[h['id']][metric])
                #         running_side = combined_block_addresses[house['id']]['side']
                #         start = j
                #         added_houses.update([h['id'] for h in new_houses])

                # # Now, sort the last side
                # houses_left = [h for h in new_houses[start:] if h['id'] not in added_houses]
                # new_house_order += sorted(houses_left, key=lambda h, metric=metric: combined_block_addresses[h['id']][metric])

            sub_block.houses = new_house_order

            # # print("BEFORE", self.depot if i == 0 else new_sub_blocks[i - 1], flush=True)
            # entrance_pt = self._calculate_entrance(self.depot if i == 0 else new_sub_blocks[i - 1].navigation_points[-1], new_houses[0])
            # if i == len(sub_blocks) - 1:
            #     exit_pt = self._calculate_entrance(self.depot, new_houses[-1])
            # elif len(sub_blocks[i + 1].houses) == 0:
            #     exit_pt = sub_blocks[i + 1].navigation_points[0]
            # else:
            #     exit_pt = self._calculate_exit(new_houses[-1], sub_blocks[i + 1].houses[0])
            # print("For last house {}, exit point {}".format(new_houses[-1], exit_pt), flush=True)
            # new_block = self._process_sub_block(houses=new_houses, block_id=block_id, entrance=entrance_pt, exit=exit_pt)

            # if type(new_block) == tuple:
            #     new_sub_blocks.extend(new_block)
            # else:
            #     new_sub_blocks.append(new_block)
            # print('New navigation points', new_block.navigation_points, sub_block.navigation_points, flush=True)

        return sub_blocks

    def post_process(self, tour: Tour) -> list[SubBlock]:
        # Iterate through the solution and add subsegments
        walk_list: list[SubBlock] = []

        # From the index of each stop, get the points for those stops
        tour_stops: list[Point] = [
            self.points[h["location"]["index"]] for h in tour["stops"]
        ]
        depot, houses = tour_stops[0], tour_stops[1:-1]
        self.depot = depot

        current_sub_block_houses: list[Point] = []

        # Take the side closest to the first house (likely where a canvasser would park)
        running_intersection = depot
        running_block_id = self.house_id_to_block_id[houses[0]["id"]]

        # Process the list
        for house, next_house in itertools.pairwise(houses):
            next_block_id = self.house_id_to_block_id[next_house["id"]]
            current_sub_block_houses.append(house)

            if next_block_id != running_block_id:
                # Calculate the entrance to the block which is ending
                entrance_pt = self._calculate_entrance(
                    running_intersection, current_sub_block_houses[0]
                )
                # Calculate the exit from the block which is ending
                exit_pt = self._calculate_exit(house, next_house)
                subsegment = self._process_sub_block(
                    current_sub_block_houses,
                    running_block_id,
                    entrance=entrance_pt,
                    exit=exit_pt,
                )

                if type(subsegment) == tuple:
                    walk_list.append(subsegment[0])
                    walk_list.append(subsegment[1])
                else:
                    walk_list.append(subsegment)

                current_sub_block_houses = []

                # After completing this segment, the canvasser is at the end of the subsegment
                running_intersection = exit_pt
                running_block_id = next_block_id

        # Since we used pairwise, the last house is never evaluated
        current_sub_block_houses.append(houses[-1])

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

        if type(sub_block) == tuple:
            walk_list.extend(sub_block)
        else:
            walk_list.append(sub_block)

        # walk_list.append(self._process_sub_block(
        #     current_sub_block_points, running_block_id, entrance=entrance_point, exit=exit_point))

        # Combine sub-blocks that are on the same block, if there are houses on the same side
        num_houses = sum([len(sub_block.houses) for sub_block in walk_list])
        # print("Number of houses before combining sub-blocks: {}".format(num_houses))
        walk_list = self.combine_sub_blocks(walk_list)
        num_houses = sum([len(sub_block.houses) for sub_block in walk_list])
        # print("Number of houses after combining sub-blocks: {}".format(num_houses))

        # Fill in any holes
        walk_list = self.fill_holes(walk_list)

        # If the final sub-block is empty, remove it
        if len(walk_list[-1].houses) == 0:
            walk_list.pop()

        return walk_list

    def generate_file(self, walk_list: list[SubBlock], output_file: str):
        """
        Generate a JSON file with the walk list (for front-end).

        Parameters
        ----------
            walk_list (list[SubBlock]): The walk list to generate the file for
            output_file (str): The file to write to
        """
        # Generate the JSON file
        list_out = {}
        list_out["blocks"] = []
        for sub_block in walk_list:
            nodes = []
            for nav_pt in sub_block.navigation_points:
                nodes.append({"lat": nav_pt["lat"], "lon": nav_pt["lon"]})
            houses = []
            for house in sub_block.houses:
                # Lookup this entry by uuid
                try:
                    house_voter_info: HousePeople = self.house_to_voters[pt_id(house)]
                except KeyError:
                    print(colored("House {} with id {} not found in voter info".format(house, pt_id(house)), "red"))
                    sys.exit(1)

                assert house_voter_info["latitude"] == house["lat"] and house_voter_info["longitude"] == house["lon"], \
                    "House coordinate mismatch: voter info file had {}, routing file had {}".format(
                        (house_voter_info["latitude"], house_voter_info["longitude"]), (house["lat"], house["lon"]))

                houses.append(
                    HouseOutput(
                        display_address=house_voter_info["display_address"],
                        city=house_voter_info["city"],
                        state=house_voter_info["state"],
                        zip=house_voter_info["zip"],
                        uuid=pt_id(house),
                        latitude=house_voter_info["latitude"],
                        longitude=house_voter_info["longitude"],
                        voter_info=house_voter_info["voter_info"],
                        subsegment_start=house["subsegment_start"],
                ))

            list_out["blocks"].append({"nodes": nodes, "houses": houses})

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

    return distance_costs["travelTimes"][(idx1 * int(num_points)) + idx2], \
        distance_costs["distances"][(idx1 * int(num_points)) + idx2]


def process_solution(
    solution: Solution, optimizer_points: list[Point], requested_blocks: blocks_file_t, viz_path: str = VIZ_PATH, problem_path: str = PROBLEM_PATH
):
    point_orders: list[list[tuple[Point, int]]] = []

    for i, route in enumerate(solution["tours"]):
        point_orders.append([])
        for stop in route["stops"][1:-1]:
            point_orders[i].append((optimizer_points[stop["location"]["index"]], stop["location"]["index"]))

    display_distance_matrix(optimizer_points, os.path.join(problem_path, "distances.json")).save(
        os.path.join(viz_path, "distances.html")
    )

    # house_dcs = [[HouseDistances.get_distance(i, j) for (i, j) in itertools.pairwise(list)] for list in point_orders]
    distances_file = os.path.join(problem_path, "distances.json")
    house_dcs = [
        [get_distance_cost(i[1], j[1], distances_file) for (i, j) in itertools.pairwise(list)]
        for list in point_orders
    ]

    points = [[i[0] for i in list] for list in point_orders]

    display_house_orders(points, dcs=house_dcs).save(
        os.path.join(viz_path, "optimal.html")
    )

    post_processor = PostProcess(requested_blocks, points=optimizer_points)
    walk_lists: list[list[SubBlock]] = []
    for i, tour in enumerate(solution["tours"]):
        # Do not count the startingg location service at the start or end
        tour["stops"] = tour["stops"][1:-1] if TURF_SPLIT else tour["stops"]

        if len(tour["stops"]) == 0:
            print(f"List {i} has 0 stops")
            continue

        walk_lists.append(post_processor.post_process(tour))

    # Save the walk lists
    display_walk_lists(walk_lists).save(
        os.path.join(viz_path, "walk_lists.html")
    )

    list_visualizations = display_individual_walk_lists(walk_lists)
    for i, walk_list in enumerate(list_visualizations):
        walk_list.save(os.path.join(viz_path, f"walk_lists_{i}.html"))

    for i in range(len(walk_lists)):
        post_processor.generate_file(
            walk_lists[i], os.path.join(viz_path, f"files_{i}.json")
        )


if __name__ == "__main__":
    # Load the requested blocks
    requested_blocks = json.load(open(requested_blocks_file, "r"))

    # Generate node distance matrix
    NodeDistances(requested_blocks)

    # Generate block distance matrix
    BlockDistances(requested_blocks)

    # Initialize calculator for mixed distances
    MixDistances()

    # Load the optimizer points from pickle
    optimizer_points = pickle.load(open(optimizer_points_pickle_file, "rb"))

    # Load the solution file
    solution: Solution = Optimizer.process_solution()

    # Process the solution
    process_solution(solution, optimizer_points, requested_blocks)
