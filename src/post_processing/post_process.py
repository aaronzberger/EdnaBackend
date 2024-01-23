import itertools
import json
import os
import sys
from copy import deepcopy

from termcolor import colored

from src.config import (
    CAMPAIGN_ID,
    SubAbode,
    Voter,
    AbodeGeography,
    Abode,
    VIZ_PATH,
    HouseOutput,
    NodeType,
    InternalPoint,
    WriteablePoint,
    generate_pt_id,
    pt_id,
    details_file,
    ABODE_DB_IDX,
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
    def __init__(self, mix_distances: MixDistances):
        self.db = Database()

        RouteMaker()

        # Map block IDs to the UUIDs on this route on that block
        self.block_to_abode_ids: dict[str, list[str]] = {}
        self.inserted_abode_ids: set[str] = set()

        self.mix_distances = mix_distances

    def _calculate_exit(
        self, final_house: InternalPoint, next_house: InternalPoint
    ) -> InternalPoint:
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
        origin_block_id = self.db.get_dict(final_house["id"], ABODE_DB_IDX)["block_id"]
        origin_block = self.db.get_dict(origin_block_id, BLOCK_DB_IDX)

        end_node = deepcopy(origin_block["nodes"][-1])
        end_node["type"] = NodeType.node

        through_end = self.mix_distances.get_distance(p1=end_node, p2=next_house)

        if through_end is not None:
            through_end += origin_block["abodes"][final_house["id"]]["distance_to_end"]

        start_node = deepcopy(origin_block["nodes"][0])
        start_node["type"] = NodeType.node

        through_start = self.mix_distances.get_distance(p1=start_node, p2=next_house)

        if through_start is not None:
            through_start += origin_block["abodes"][final_house["id"]][
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

    def _calculate_entrance(
        self, intersection: InternalPoint, next_house: InternalPoint
    ) -> InternalPoint:
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
        destination_block_id = self.db.get_dict(next_house["id"], ABODE_DB_IDX)[
            "block_id"
        ]
        destination_block = self.db.get_dict(destination_block_id, BLOCK_DB_IDX)

        intersection["type"] = NodeType.node
        end_node = deepcopy(destination_block["nodes"][-1])
        end_node["type"] = NodeType.node

        through_end = self.mix_distances.get_distance(p1=intersection, p2=end_node)

        if through_end is not None:
            through_end += destination_block["abodes"][next_house["id"]][
                "distance_to_end"
            ]

        start_node = deepcopy(destination_block["nodes"][0])
        start_node["type"] = NodeType.node

        through_start = self.mix_distances.get_distance(p1=intersection, p2=start_node)

        if through_start is not None:
            through_start += destination_block["abodes"][next_house["id"]][
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
        self, sub_block: SubBlock, entrance: InternalPoint, exit: InternalPoint
    ) -> tuple[SubBlock, SubBlock]:
        nav_pts_1 = sub_block.nodes[: len(sub_block.nodes) // 2 + 1]
        nav_pts_2 = sub_block.nodes[len(sub_block.nodes) // 2:]

        assert nav_pts_1[-1] == nav_pts_2[0]

        abodes_1: list[SubAbode] = [
            i for i in sub_block.abodes if i["subsegment_start"] < len(nav_pts_1) - 1
        ]
        abodes_2: list[SubAbode] = [
            i for i in sub_block.abodes if i["subsegment_start"] >= len(nav_pts_1) - 1
        ]

        for i in range(len(abodes_2)):
            abodes_2[i]["subsegment_start"] -= len(nav_pts_1) - 1

        sub_block_1 = SubBlock(
            block_id=sub_block.block_id,
            nodes=nav_pts_1,
            abodes=abodes_1,
        )

        sub_block_2 = SubBlock(
            block_id=sub_block.block_id,
            nodes=nav_pts_2,
            abodes=abodes_2,
        )

        return sub_block_1, sub_block_2

    def _process_sub_block(
        self,
        abode_points: list[InternalPoint],
        block_id: str,
        entrance: InternalPoint,
        exit: InternalPoint,
        verbose=False,
    ) -> SubBlock | tuple[SubBlock, SubBlock]:
        """
        Process a single sub-block.

        This function takes a list of houses on a single block and returns a SubBlock object
        which contains the navigation points, houses, and other information needed to use the route.

        Parameters
        ----------
            houses (list[Point]): The houses to process
            block_id (str): The block ID of the block to process
            entrance (Point): The entrance to the block
            exit (Point): The exit to the block
            verbose (bool, optional): Whether to print out debug information. Defaults to False.

        Returns
        -------
            SubBlock | tuple[SubBlock, SubBlock]: The processed sub-block, or two sub-blocks if the block was split
        """
        abode_ids = [i["id"] for i in abode_points]
        all_abode_geographies_on_block = {
            id: geography
            for id, geography in self.db.get_dict(block_id, BLOCK_DB_IDX)["abodes"].items()
            if id in abode_ids
        }

        if verbose:
            if len(abode_ids) != len(self.block_to_abode_ids[block_id]):
                print(
                    colored(
                        f"Notice: There are {len(abode_ids)} houses, but {len(self.block_to_abode_ids[block_id])} houses on this block in the route"
                        + f"\nThe unplaced houses are {set(self.block_to_abode_ids[block_id]) - set(abode_ids)}",
                        "blue",
                    )
                )

        block = self.db.get_dict(block_id, BLOCK_DB_IDX)

        extremum: tuple[InternalPoint, InternalPoint] = (entrance, exit)

        # region: Calculate the navigation points
        if pt_id(entrance) != pt_id(exit):
            # The canvasser enters at one point and exits at another, so simply
            # order the navigation points to align with the rest of the route
            navigation_points = (
                block["nodes"]
                if pt_id(entrance) == pt_id(block["nodes"][0])
                else block["nodes"][::-1]
            )

        elif pt_id(entrance) == pt_id(exit) == pt_id(block["nodes"][0]):
            # The canvasser enters at the same point they exit (the start). Thus, this is an
            # "out-and-back" block, so let's find the furthest point the canvasser will walk to
            # (the end_extremum)
            extremum_abode_id, extremum_abode = max(
                all_abode_geographies_on_block.items(), key=lambda a: a[1]["distance_to_start"]
            )
            end_extremum = project_to_line(
                p1=InternalPoint(
                    lat=extremum_abode["point"]["lat"],
                    lon=extremum_abode["point"]["lon"],
                    type=NodeType.house,
                    id=extremum_abode_id,
                ),
                p2=block["nodes"][extremum_abode["subsegment_start"]],
                p3=block["nodes"][extremum_abode["subsegment_end"]],
            )
            extremum = (extremum[0], end_extremum)

            # The navigation points will go out, then back
            navigation_points = (
                block["nodes"][: extremum_abode["subsegment_start"] + 1]
                + [end_extremum]
                + list(reversed(block["nodes"][: extremum_abode["subsegment_start"] + 1]))
            )

        elif pt_id(entrance) == pt_id(exit) == pt_id(block["nodes"][-1]):
            # The canvasser enters at the same point they exit (the end). So, do the same
            # but in reverse, calculating the furthest point they go towards the start (the start_extremum)
            extremum_abode_id, extremum_abode = max(
                all_abode_geographies_on_block.items(), key=lambda a: a[1]["distance_to_end"]
            )
            start_extremum = project_to_line(
                p1=InternalPoint(
                    lat=extremum_abode["point"]["lat"],
                    lon=extremum_abode["point"]["lon"],
                    type=NodeType.house,
                    id=extremum_abode_id,
                ),
                p2=block["nodes"][extremum_abode["subsegment_start"]],
                p3=block["nodes"][extremum_abode["subsegment_end"]],
            )
            extremum = (start_extremum, extremum[1])

            # The navigation points will go out, then back
            navigation_points = (
                list(reversed(block["nodes"][extremum_abode["subsegment_end"]:]))
                + [start_extremum]
                + block["nodes"][extremum_abode["subsegment_end"]:]
            )

        else:
            print(
                colored(
                    'Error: "entrance" and "exit" were not valid for this block', "red"
                )
            )
        # endregion

        # Find the abodes which are on this route but not on this block.
        # TODO/NOTE: If the optimizer is good enough (further, the distance matrix is good enough),
        # then this shouldn't be needed. For now, it will result in less confusing routes since the
        # distance matrix doesn't do street crossings amazingly across blocks and intersections.
        unused_abodes = (
            set(self.block_to_abode_ids[block_id])
            - set(abode_ids)
            - self.inserted_abode_ids
        )

        sub_abodes: list[SubAbode] = []

        # region: Order the abodes
        if pt_id(entrance) != pt_id(exit):
            running_side = all_abode_geographies_on_block[abode_points[0]["id"]]["side"]
            start = 0
            i = 0
            new_abode_order: list[InternalPoint] = []
            running_distance_to_start = 0
            added_abode_ids = set()

            metric = (
                "distance_to_start"
                if pt_id(entrance) == pt_id(block["nodes"][0])
                else "distance_to_end"
            )

            for i, abode_point in enumerate(abode_points):
                if all_abode_geographies_on_block[abode_point["id"]]["side"] != running_side:
                    # Add any abodes on the same side with less distance to the start
                    new_abode_points = deepcopy(abode_points[start:i])
                    new_abode_points = [a for a in new_abode_points if a["id"] not in added_abode_ids]

                    running_distance_to_start = max(
                        running_distance_to_start,
                        max(
                            [
                                all_abode_geographies_on_block[abode["id"]][metric]
                                for abode in new_abode_points + new_abode_order
                            ]
                        ),
                    )

                    for remaining_abode_point in abode_points[i + 1:]:
                        if (
                            all_abode_geographies_on_block[remaining_abode_point["id"]]["side"] == running_side
                            and all_abode_geographies_on_block[remaining_abode_point["id"]][metric]
                            < running_distance_to_start
                            and remaining_abode_point["id"] not in added_abode_ids
                        ):
                            new_abode_points.append(remaining_abode_point)

                    # Also, add any abodes on future blocks which can be moved up to this block
                    for potential_abode_id in unused_abodes:
                        abode_geography: AbodeGeography = self.db.get_dict(block_id, BLOCK_DB_IDX)[
                            "abodes"
                        ][potential_abode_id]
                        if (
                            abode_geography["side"] != running_side
                            and abode_geography[metric] < running_distance_to_start
                            and potential_abode_id not in added_abode_ids
                        ):
                            new_abode_points.append(
                                InternalPoint(
                                    lat=abode_geography["lat"],
                                    lon=abode_geography["lon"],
                                    type=NodeType.house,
                                    id=potential_abode_id,
                                )
                            )

                            abode_ids.append(potential_abode_id)
                            all_abode_geographies_on_block[potential_abode_id] = abode_geography

                            if verbose:
                                print(
                                    f"Moving up abode with id {potential_abode_id} to this block (different entrance/exit)"
                                )

                    new_abode_order += sorted(
                        new_abode_points,
                        key=lambda h, metric=metric: all_abode_geographies_on_block[h["id"]][metric],
                    )
                    running_side = all_abode_geographies_on_block[abode_point["id"]]["side"]
                    start = i
                    added_abode_ids.update([h["id"] for h in new_abode_points])

            # Now, sort the last side
            abodes_left = [h for h in abode_points[start:] if h["id"] not in added_abode_ids]

            # Also, add any abodes on future blocks which can be moved up to this block
            for potential_abode_id in unused_abodes:
                abode_geography: AbodeGeography = self.db.get_dict(block_id, BLOCK_DB_IDX)[
                    "abodes"
                ][potential_abode_id]
                # info = self.requested_blocks[block_id]["addresses"][potential_abode_id]
                if (
                    abode_geography["side"] == running_side
                    and potential_abode_id not in added_abode_ids
                ):
                    abodes_left.append(
                        InternalPoint(
                            lat=abode_geography["lat"],
                            lon=abode_geography["lon"],
                            type=NodeType.house,
                            id=potential_abode_id,
                        )
                    )

                    abode_ids.append(potential_abode_id)
                    all_abode_geographies_on_block[potential_abode_id] = abode_geography

                    if verbose:
                        print(
                            f"Moving up abode with id {potential_abode_id} to this block (different entrance/exit)"
                        )

            new_abode_order += sorted(
                abodes_left, key=lambda h: all_abode_geographies_on_block[h["id"]][metric]
            )

            # TODO: Should be able to remove this and just use new_abode_order below?
            abode_points = new_abode_order

            # We're always going forward, so the subsegments are as they are
            for i, abode_point in enumerate(abode_points):
                sub_start = block["nodes"][all_abode_geographies_on_block[abode_point["id"]]["subsegment_start"]]
                sub_end = block["nodes"][all_abode_geographies_on_block[abode_point["id"]]["subsegment_end"]]
                sub_start_idx = navigation_points.index(sub_start)
                sub_end_idx = navigation_points.index(sub_end)

                abode: Abode = self.db.get_dict(abode_point["id"], ABODE_DB_IDX)
                subsegment_start = min(sub_start_idx, sub_end_idx)

                sub_abodes.append(SubAbode(
                    abode_id=abode_point["id"],
                    point=WriteablePoint(lat=abode_point["lat"], lon=abode_point["lon"]),
                    distance_to_start=all_abode_geographies_on_block[abode_point["id"]]["distance_to_start"],
                    distance_to_end=all_abode_geographies_on_block[abode_point["id"]]["distance_to_end"],
                    side=all_abode_geographies_on_block[abode_point["id"]]["side"],
                    distance_to_road=all_abode_geographies_on_block[abode_point["id"]]["distance_to_road"],
                    subsegment_start=subsegment_start,
                    subsegment_end=subsegment_start + 1,
                    display_address=abode["display_address"],
                    voter_ids=abode["voter_ids"],
                    block_id=abode["block_id"],
                    city=abode["city"],
                    state=abode["state"],
                    zip=abode["zip"],
                ))

        elif pt_id(entrance) == pt_id(exit):
            # The optimal path is always to go out on one side and back on the other
            # (since you must go out and back anyway, and this minimizes street crossings)
            out_side = [
                h
                for h in abode_points
                if all_abode_geographies_on_block[h["id"]]["side"]
                == all_abode_geographies_on_block[abode_points[0]["id"]]["side"]
            ]
            back_side = [
                h
                for h in abode_points
                if all_abode_geographies_on_block[h["id"]]["side"]
                != all_abode_geographies_on_block[abode_points[0]["id"]]["side"]
            ]

            # To avoid mis-placing abodes off the end, sort by the distance to the entrance/exit
            metric_out = (
                "distance_to_start"
                if pt_id(entrance) == pt_id(block["nodes"][0])
                else "distance_to_end"
            )

            # Put the "out" side abodes first, then the "back" side abodes
            abode_points = sorted(
                out_side,
                key=lambda h: all_abode_geographies_on_block[h["id"]][metric_out],
            ) + sorted(
                back_side, key=lambda h: all_abode_geographies_on_block[h["id"]][metric_out], reverse=True
            )

            # For the out abodes, we're always going forward, so the subsegments are as they are
            for i, abode_point in enumerate(out_side):
                out_nav_nodes = navigation_points[: len(navigation_points) // 2 + 1]
                sub_start = block["nodes"][all_abode_geographies_on_block[abode_point["id"]]["subsegment_start"]]
                sub_end = block["nodes"][all_abode_geographies_on_block[abode_point["id"]]["subsegment_end"]]

                if verbose:
                    print(
                        f'Out nav nodes are {out_nav_nodes} for abode {abode_point["id"]}, subsegment {sub_start}, {sub_end}'
                    )

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
                            f"WARNNING: abode {abode_point['id']} 's subsegment is no longer existent: extremum calculation likely failed",
                            "yellow",
                        )
                    )

                abode: Abode = self.db.get_dict(abode_point["id"], ABODE_DB_IDX)
                subsegment_start = min(sub_start_idx, sub_end_idx)

                sub_abodes.append(SubAbode(
                    abode_id=abode_point["id"],
                    point=WriteablePoint(lat=abode_point["lat"], lon=abode_point["lon"]),
                    distance_to_start=all_abode_geographies_on_block[abode_point["id"]]["distance_to_start"],
                    distance_to_end=all_abode_geographies_on_block[abode_point["id"]]["distance_to_end"],
                    side=all_abode_geographies_on_block[abode_point["id"]]["side"],
                    distance_to_road=all_abode_geographies_on_block[abode_point["id"]]["distance_to_road"],
                    subsegment_start=subsegment_start,
                    subsegment_end=subsegment_start + 1,
                    display_address=abode["display_address"],
                    voter_ids=abode["voter_ids"],
                    block_id=abode["block_id"],
                    city=abode["city"],
                    state=abode["state"],
                    zip=abode["zip"],
                ))

            # For the back abodes, they are on the second half of the subsegments
            for i, abode_point in enumerate(back_side):
                back_nav_nodes = navigation_points[len(navigation_points) // 2:]
                sub_start = block["nodes"][all_abode_geographies_on_block[abode_point["id"]]["subsegment_start"]]
                sub_end = block["nodes"][all_abode_geographies_on_block[abode_point["id"]]["subsegment_end"]]

                if verbose:
                    print(
                        f'Back nav nodes are {back_nav_nodes} for abode {abode_point["id"]}, subsegment {sub_start}, {sub_end}'
                    )

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
                            f"WARNNING: abode {abode_point['id']} 's subsegment is no longer existent: extremum calculation likely failed",
                            "yellow",
                        )
                    )

                subsegment_start = min(sub_start_idx, sub_end_idx) + len(navigation_points) // 2 - 1

                abode: Abode = self.db.get_dict(abode_point["id"], ABODE_DB_IDX)

                sub_abodes.append(SubAbode(
                    abode_id=abode_point["id"],
                    point=WriteablePoint(lat=abode_point["lat"], lon=abode_point["lon"]),
                    distance_to_start=all_abode_geographies_on_block[abode_point["id"]]["distance_to_start"],
                    distance_to_end=all_abode_geographies_on_block[abode_point["id"]]["distance_to_end"],
                    side=all_abode_geographies_on_block[abode_point["id"]]["side"],
                    distance_to_road=all_abode_geographies_on_block[abode_point["id"]]["distance_to_road"],
                    subsegment_start=subsegment_start,
                    subsegment_end=subsegment_start + 1,
                    display_address=abode["display_address"],
                    voter_ids=abode["voter_ids"],
                    block_id=abode["block_id"],
                    city=abode["city"],
                    state=abode["state"],
                    zip=abode["zip"],
                ))
        # endregion

        sub_block = SubBlock(
            block_id=block_id,
            nodes=navigation_points,
            abodes=sub_abodes,
        )

        if pt_id(entrance) == pt_id(exit):
            new_dual = self._split_sub_block(sub_block, entrance, exit)
            return new_dual

        for uuid in abode_ids:
            self.inserted_abode_ids.add(uuid)

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

        for first, second in itertools.pairwise(walk_list):
            new_walk_list.append(first)

            # If the end of the first subblock is not the same as the start of the second subblock, add a new subblock
            if pt_id(first.nodes[-1]) != pt_id(second.nodes[0]):
                route_info = RouteMaker.get_route(first.nodes[-1], second.nodes[0])

                if route_info is None:
                    print(
                        colored(
                            "Not adding route between sub-blocks because no route was found",
                        ),
                        "yellow",
                    )
                    continue
                block_ids, distance = route_info

                running_node = first.nodes[-1]
                for block_id in block_ids:
                    # TODO: Check if we can remove this. It is the only reference to all_blocks
                    # May be able to just refer to block_id in the subblock
                    block = self.db.get_dict(block_id, BLOCK_DB_IDX)

                    start = InternalPoint(
                        lat=block["nodes"][0]["lat"],
                        lon=block["nodes"][0]["lon"],
                        type=NodeType.node,
                        id=generate_pt_id(
                            block["nodes"][0]["lat"], block["nodes"][0]["lon"]
                        ),
                    )
                    end = InternalPoint(
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
                            block_id=block_id,
                            nodes=nav_pts,
                            abodes=[],
                        )
                    )

        new_walk_list.append(walk_list[-1])

        return new_walk_list

    def post_process(self, route: list[InternalPoint]) -> list[SubBlock]:
        """
        Post-process a single route.

        This process maps a raw route outputted by the optimizer (a list of geographic points)
        to a canvassing route (a list of sub-blocks), which is needed for using the route.

        Parameters
        ----------
            route (list[Point]): The route to post-process

        Returns
        -------
            list[SubBlock]: The post-processed route
        """
        # Iterate through the solution and add subsegments
        walk_list: list[SubBlock] = []

        # region Full route pre-processing
        depot_point, abode_points = route[0], route[1:-1]
        self.depot = depot_point

        # Fill in the mapping from block ids to abode ids
        for abode_point in abode_points:
            if (
                self.db.get_dict(abode_point["id"], ABODE_DB_IDX)["block_id"]
                not in self.block_to_abode_ids
            ):
                self.block_to_abode_ids[
                    self.db.get_dict(abode_point["id"], ABODE_DB_IDX)["block_id"]
                ] = [abode_point["id"]]
            else:
                self.block_to_abode_ids[
                    self.db.get_dict(abode_point["id"], ABODE_DB_IDX)["block_id"]
                ].append(abode_point["id"])
        # endregion

        # region SubBlock processing
        current_sub_block_abode_points: list[InternalPoint] = []

        # Take the side closest to the first house (likely where a canvasser would park)
        running_intersection = depot_point
        running_block_id = self.db.get_dict(abode_points[0]["id"], ABODE_DB_IDX)["block_id"]

        for abode_point, next_abode_point in itertools.pairwise(abode_points):
            next_block_id = self.db.get_dict(next_abode_point["id"], ABODE_DB_IDX)["block_id"]

            if abode_point["id"] not in self.inserted_abode_ids:
                current_sub_block_abode_points.append(abode_point)

            # Process the end of a SubBlock (and add it)
            if next_block_id != running_block_id:
                if len(current_sub_block_abode_points) == 0:
                    running_block_id = next_block_id
                    continue

                # Calculate the entrance to the block which is ending
                entrance_pt = self._calculate_entrance(
                    running_intersection, current_sub_block_abode_points[0]
                )

                # Calculate the exit from the block which is ending
                exit_pt = self._calculate_exit(abode_point, next_abode_point)
                sub_block_or_blocks = self._process_sub_block(
                    current_sub_block_abode_points,
                    running_block_id,
                    entrance=entrance_pt,
                    exit=exit_pt,
                )

                if isinstance(sub_block_or_blocks, SubBlock):
                    walk_list.append(sub_block_or_blocks)
                elif isinstance(sub_block_or_blocks, tuple):
                    walk_list.extend(sub_block_or_blocks)
                else:
                    raise ValueError(
                        f"Unexpected return type from _process_sub_block: {type(sub_block_or_blocks)}"
                    )

                current_sub_block_abode_points = []

                # After completing this segment, the canvasser is at the end of the subsegment
                running_intersection = exit_pt
                running_block_id = next_block_id

        # Since we used pairwise, the last house is never evaluated
        if abode_points[-1]["id"] not in self.inserted_abode_ids:
            current_sub_block_abode_points.append(abode_points[-1])

        if len(current_sub_block_abode_points) != 0:
            # Determine the final intersection the canvasser will end up at to process the final subsegment
            exit_point = self._calculate_entrance(depot_point, current_sub_block_abode_points[-1])
            entrance_point = self._calculate_entrance(
                running_intersection, current_sub_block_abode_points[0]
            )

            sub_block_or_blocks = self._process_sub_block(
                current_sub_block_abode_points,
                running_block_id,
                entrance=entrance_point,
                exit=exit_point,
            )

            if isinstance(sub_block_or_blocks, SubBlock):
                walk_list.append(sub_block_or_blocks)
            elif isinstance(sub_block_or_blocks, tuple):
                walk_list.extend(sub_block_or_blocks)
            else:
                raise ValueError(
                    f"Unexpected return type from _process_sub_block: {type(sub_block_or_blocks)}"
                )
        # endregion

        # region Full route post-processing
        # Fill in any holes
        walk_list = self.fill_holes(walk_list)

        # If the final sub-block is empty, remove it
        if len(walk_list[-1].abodes) == 0:
            walk_list.pop()
        # endregion

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
            for nav_pt in sub_block.nodes:
                nodes.append({"lat": nav_pt["lat"], "lon": nav_pt["lon"]})
            houses = []
            for abode in sub_block.abodes:
                houses.append(
                    HouseOutput(
                        display_address=abode["display_address"],
                        city=abode["city"],
                        state=abode["state"],
                        zip=abode["zip"],
                        uuid=pt_id(abode["point"]),
                        latitude=abode["point"]["lat"],
                        longitude=abode["point"]["lon"],
                        voter_info=abode["voter_ids"],
                        subsegment_start=abode["subsegment_start"],
                    )
                )

            list_out["blocks"].append({"nodes": nodes, "houses": houses})

        list_out["form"] = form

        # Write the file
        json.dump(list_out, open(output_file, "w"))


def process_partitioned_solution(
    route_parts: list[list[list[InternalPoint]]],
    mix_distances: list[MixDistances],
    viz_path: str = VIZ_PATH,
    id: str = CAMPAIGN_ID,
):
    # Display the house orders together
    combined_routes: list[list[InternalPoint]] = [
        route for sublist in route_parts for route in sublist
    ]
    display_house_orders(combined_routes).save(
        os.path.join(viz_path, "direct_output.html")
    )

    if os.path.exists(details_file):
        details = json.load(open(details_file))
    else:
        details = {}

    processed_routes: list[list[SubBlock]] = []
    for i, part in enumerate(route_parts):
        for j, route in enumerate(part):
            # Process the route
            post_processor = PostProcess(mix_distances[i])
            processed_routes.append(post_processor.post_process(route))

            list_id = f"{CAMPAIGN_ID}-{id}-{i}-{j}"
            if list_id in details:
                print(colored(f"Warning: List {list_id} already exists in details"))

            num_houses = sum([len(sub_block.abodes) for sub_block in processed_routes[-1]])
            distance = 0
            for sub_block in processed_routes[-1]:
                distance += sub_block.length
            start_point = {
                "lat": processed_routes[-1][0].nodes[0]["lat"],
                "lon": processed_routes[-1][0].nodes[0]["lon"],
            }

            details[list_id] = {
                "distance": distance,
                "num_houses": num_houses,
                "start_point": start_point,
            }

    json.dump(details, open(details_file, "w"))

    print(f'There are {len(processed_routes)} lists in this solution, and the first has {len(processed_routes[0])} sub-blocks')

    list_visualizations = display_individual_walk_lists(processed_routes)
    for i, walk_list in enumerate(list_visualizations):
        print('Saving to', os.path.join(viz_path, f"{CAMPAIGN_ID}-{id}-{i}.html"))
        walk_list.save(os.path.join(viz_path, f"{CAMPAIGN_ID}-{id}-{i}.html"))

    form = json.load(
        open(os.path.join("regions", CAMPAIGN_ID, "input", "form.json"), "r")
    )

    for i in range(len(processed_routes)):
        list_id = f"{CAMPAIGN_ID}-{id}-{i}"
        post_processor.generate_file(
            processed_routes[i],
            os.path.join(viz_path, f"{list_id}.json"),
            id=list_id,
            form=form,
        )


def process_solution(
    routes: list[list[InternalPoint]],
    # optimizer_points: list[Point],
    # abode_ids: set[str],
    mix_distances: MixDistances,
    viz_path: str = VIZ_PATH,
    id: str = CAMPAIGN_ID,
):
    # house_dcs = [[HouseDistances.get_distance(i, j) for (i, j) in itertools.pairwise(list)] for list in point_orders]

    display_house_orders(routes, dcs=None).save(
        os.path.join(viz_path, "direct_output.html")
    )

    if os.path.exists(details_file):
        details = json.load(open(details_file))
    else:
        details = {}

    post_processor = PostProcess(mix_distances=mix_distances)
    routes: list[list[SubBlock]] = []
    for i, route in enumerate(routes):
        # Process the route
        routes.append(post_processor.post_process(route))

        list_id = f"{CAMPAIGN_ID}-{id}-{i}"
        if list_id in details:
            print(colored(f"Warning: List {list_id} already exists in details"))

        num_houses = sum([len(sub_block.abodes) for sub_block in routes[-1]])
        distance = 0
        for sub_block in routes[-1]:
            distance += sub_block.length
        start_point = {
            "lat": routes[-1][0].nodes[0]["lat"],
            "lon": routes[-1][0].nodes[0]["lon"],
        }

        details[list_id] = {
            "distance": distance,
            "num_houses": num_houses,
            "start_point": start_point,
        }

    json.dump(details, open(details_file, "w"))

    # all_list_abodes = set()
    # for walk_list in routes:
    #     for sub_block in walk_list:
    #         for house in sub_block.houses:
    #             all_list_abodes.add(house["id"])

    # for id in all_list_abodes:
    #     assert id in abode_ids, f"Abode {id} not in universe"

    list_visualizations = display_individual_walk_lists(routes)
    for i, walk_list in enumerate(list_visualizations):
        walk_list.save(os.path.join(viz_path, f"{CAMPAIGN_ID}-{id}-{i}.html"))

    form = json.load(
        open(os.path.join("regions", CAMPAIGN_ID, "input", "form.json"), "r")
    )

    for i in range(len(routes)):
        list_id = f"{CAMPAIGN_ID}-{id}-{i}"
        post_processor.generate_file(
            routes[i],
            os.path.join(viz_path, f"{list_id}.json"),
            id=list_id,
            form=form,
        )
