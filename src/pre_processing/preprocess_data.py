"""
Process the data from OSM. Take in overpass.json and write block_output.json
"""

import json
import pprint
from collections import defaultdict
from typing import TypedDict, TypeGuard, Literal, Optional, Tuple, NewType, cast, Self

from src.config import block_output_file, overpass_file

pp = pprint.PrettyPrinter(indent=4)

NodeId = NewType("NodeId", int)


class Node(TypedDict):
    type: Literal["node"] | Literal["way"]
    id: NodeId
    lat: float
    lon: float


Tags = NewType("Tags", dict[str, str])


class Way(TypedDict):
    type: str
    id: NodeId
    nodes: list[NodeId]
    tags: Tags


Element = Node | Way


class Model(TypedDict):
    elements: list[Element]


BlockWay = tuple[NodeId, Tags]


class Block(TypedDict):
    nodes: list[NodeId]
    ways: list[BlockWay]


BlockDef = Tuple[NodeId, int, Optional[Block]]

NodeInfo = Tuple[Node, list[Way]]


def is_node(elem: Element) -> TypeGuard[Node]:
    return elem["type"] == "node"


def is_way(elem: Element) -> TypeGuard[Way]:
    return elem["type"] == "way"


class Preprocessor:
    def __init__(
        self,
        grouped_response: dict[NodeId, NodeInfo],
        intersection_nodes: dict[NodeId, NodeInfo],
    ):
        self.grouped_response = grouped_response
        self.intersection_nodes = intersection_nodes
        self.blocks: dict[NodeId, list[BlockDef]] = defaultdict(list)
        self.explored_space: dict[NodeId, list[list[NodeId]]] = defaultdict(list)

    @staticmethod
    def group_response(response: Model) -> dict[NodeId, NodeInfo]:
        grouped_response: dict[NodeId, NodeInfo] = {}

        for element in response["elements"]:
            if is_node(element):
                grouped_response[element["id"]] = (element, [])

        last_node: Optional[NodeId] = None
        for element in response["elements"]:
            if is_node(element):
                last_node = element["id"]
            elif is_way(element):
                assert isinstance(last_node, int)
                grouped_response[cast(NodeId, last_node)][1].append(element)
            else:
                raise ValueError(element["type"])

        return grouped_response

    @classmethod
    def from_response(cls, response) -> Self:
        # Groups the JSON in a more manageable way because I couldn't figure out how to make Overpass nest the data
        grouped_response = cls.group_response(response)

        # Find the intersections based on how many directions a node branches off in
        # (1 means dead end, 3+ means intersection, 2 is just a regular road)
        intersection_nodes: dict[NodeId, NodeInfo] = {}

        # json.dump(grouped_response, open("grouped_response.json", "w"))

        for key in grouped_response:
            branches = 0
            for way in grouped_response[key][1]:
                if key == 9768363077:
                    print("among us")
                    print(way["nodes"])
                    print(key)
                index = way["nodes"].index(key)
                count = len(way["nodes"])

                if count <= 1:
                    continue

                if index == 0:
                    branches += 1
                elif index == count - 1:
                    branches += 1
                else:
                    branches += 2
            if branches == 1 or branches >= 3:
                intersection_nodes[key] = grouped_response[key]

        return cls(grouped_response, intersection_nodes)

    def bidirectional_search(self):
        for key in self.intersection_nodes:
            if key in self.explored_space:
                explored_direction_nodes = [x[-2] for x in self.explored_space[key]]
            else:
                explored_direction_nodes = None

            for way in self.intersection_nodes[key][1]:
                index = way["nodes"].index(key)
                node_count = len(way["nodes"])

                # If we're not at the end, look forwards
                if index < node_count - 1:
                    node_count = self.search(
                        explored_direction_nodes,
                        index,
                        key,
                        node_count,
                        way,
                        to_explore=way["nodes"][index + 1],
                    )

                # Similarly, if we're not at the start, look backwards
                if index > 0:
                    self.search(
                        explored_direction_nodes,
                        index,
                        key,
                        node_count,
                        way,
                        to_explore=way["nodes"][index - 1],
                        forwards=False,
                    )

    def try_advance(
        self,
        current_node: NodeId,
        current_way: Way,
        block_ways: list[BlockWay],
    ):
        # If we haven't grouped it, disregard it
        if current_node not in self.grouped_response:
            return None

        # Otherwise, take the neighbors
        other_ways = [
            x
            for x in self.grouped_response[current_node][1]
            if x["id"] != current_way["id"]
        ]

        if len(other_ways) <= 0:
            return None

        # If we have any neighbors, advance to it
        current_way = other_ways[0]
        block_ways.append((current_way["id"], current_way["tags"]))

        return current_way, len(current_way["nodes"])

    def search(
        self,
        explored_direction_nodes: Optional[list[NodeId]],
        index: int,
        node: NodeId,
        node_count: int,
        way: Way,
        to_explore: NodeId,
        forwards=True,
    ) -> int:
        if (
            explored_direction_nodes is not None
            and to_explore in explored_direction_nodes
        ):
            return node_count

        block_nodes: list[NodeId] = [node]
        block_ways: list[BlockWay] = [(way["id"], way["tags"])]

        end_found = False
        flipped = False
        index_modifier = 1 if forwards else -1

        while not end_found:
            index += index_modifier * (1 if not flipped else -1)

            if index >= node_count or index < 0:
                if (res := self.try_advance(node, way, block_ways)) is None:
                    break

                (way, node_count) = res

                flipped = way["nodes"][0] != node
                index = way["nodes"].index(node) if flipped else 1

                if not forwards:
                    flipped = not flipped

            node = way["nodes"][index]
            block_nodes.append(node)
            end_found = self.try_save_block(
                block_nodes,
                block_ways,
                node,
            )

        return node_count

    def try_save_block(
        self,
        block_nodes: list[NodeId],
        block_ways: list[BlockWay],
        current_node: NodeId,
    ) -> bool:
        if current_node not in self.intersection_nodes:
            return False

        similar_blocks = len(
            [x[0] for x in self.blocks[block_nodes[0]] if x[0] == current_node]
        )

        self.blocks[block_nodes[0]].append(
            (
                current_node,
                similar_blocks,
                {"nodes": block_nodes, "ways": block_ways},
            )
        )

        self.blocks[current_node].append((block_nodes[0], 0, None))
        self.explored_space[current_node].append(block_nodes)

        return True


def main() -> None:
    response: Model = json.load(open(overpass_file))

    p = Preprocessor.from_response(response)

    p.bidirectional_search()

    json.dump(p.blocks, open(block_output_file, "w", encoding="utf-8"), indent=4)


if __name__ == "__main__":
    main()
