import json
import os
import pprint

from src.config import BASE_DIR, block_output_file

pp = pprint.PrettyPrinter(indent=4)

response = json.load(open(os.path.join(BASE_DIR, 'input', 'squirrel_hill.json'), 'r'))

# Groups the JSON in a more manageable way because I couldn't figure out how to make Overpass nest the data

grouped_response = {}

for element in response['elements']:
    if element['type'] == 'node':
        grouped_response[element['id']] = (element, [])

for element in response['elements']:
    if element['type'] == 'node':
        lastNode = element['id']
    if element['type'] == 'way':
        grouped_response[lastNode][1].append(element)

# # Writing cleaned data to a file
# clean_data = json.dumps(grouped_response)
# file = open('Output/clean_data.json', 'w')
# file.write(clean_data)
# file.close()

# Find the intersections based on how many directions a node branches off in
# (1 means dead end, 3+ means intersection, 2 is just a regular road)

intersection_nodes = {}

for key in grouped_response:
    branches = 0
    for way in grouped_response[key][1]:
        index = way['nodes'].index(key)
        count = len(way['nodes'])
        if count > 1:
            if index == 0:
                branches += 1
            elif index == count - 1:
                branches += 1
            else:
                branches += 2
    if branches == 1 or branches >= 3:
        intersection_nodes[key] = grouped_response[key]

# # Writing instersection nodes to a file
# intersection_json = json.dumps(intersection_nodes)
# file = open('Output/intersectionnodes.json', 'w')
# file.write(intersection_json)
# file.close()

blocks = {}
explored_space = {}

for key in intersection_nodes:

    if key in explored_space:
        explored_direction_nodes = [x[-2] for x in explored_space[key]]
    else:
        explored_direction_nodes = None

    for way in intersection_nodes[key][1]:
        index = way['nodes'].index(key)
        node_count = len(way['nodes'])

        # look forwards
        if index < node_count - 1:
            next_node = way['nodes'][index + 1]
            if (
                explored_direction_nodes is None
                or next_node not in explored_direction_nodes
            ):
                end_found = False
                current_node = key
                block_nodes = [key]
                current_way = way
                block_ways = [(way['id'], way['tags'])]
                local_index = index
                flipped = False
                while not end_found:
                    local_index += 1 if not flipped else -1
                    if local_index >= node_count or local_index < 0:
                        if current_node in grouped_response:
                            otherWays = [
                                x
                                for x in grouped_response[current_node][1]
                                if x['id'] != current_way['id']
                            ]
                            if len(otherWays) > 0:
                                current_way = otherWays[0]
                                block_ways.append((current_way['id'], current_way['tags']))
                                node_count = len(current_way['nodes'])
                            else:
                                break
                        else:
                            break
                        flipped = current_way['nodes'].index(current_node) != 0
                        local_index = (
                            current_way['nodes'].index(current_node) if flipped else 1
                        )
                    current_node = current_way['nodes'][local_index]
                    block_nodes.append(current_node)
                    if current_node in intersection_nodes:
                        # save block
                        if block_nodes[0] in blocks:
                            similar_blocks = len([x[0] for x in blocks[block_nodes[0]] if x[0] == current_node])

                            blocks[block_nodes[0]].append(
                                (current_node, similar_blocks, {'nodes': block_nodes, 'ways': block_ways})
                            )
                        else:
                            blocks[block_nodes[0]] = [
                                (current_node, 0, {'nodes': block_nodes, 'ways': block_ways})
                            ]

                        if current_node in blocks:
                            blocks[current_node].append((block_nodes[0], 0, None))
                        else:
                            blocks[current_node] = [(block_nodes[0], 0, None)]

                        if current_node in explored_space:
                            explored_space[current_node].append(block_nodes)
                        else:
                            explored_space[current_node] = [block_nodes]
                        end_found = True
        # look backwards
        if index > 0:
            previous_node = way['nodes'][index - 1]
            if (
                explored_direction_nodes is None
                or previous_node not in explored_direction_nodes
            ):
                end_found = False
                current_node = key
                block_nodes = [key]
                current_way = way
                block_ways = [(way['id'], way['tags'])]
                local_index = index
                flipped = False
                while not end_found:
                    local_index -= 1 if not flipped else -1
                    if local_index >= node_count or local_index < 0:
                        if current_node in grouped_response:
                            otherWays = [
                                x
                                for x in grouped_response[current_node][1]
                                if x['id'] != current_way['id']
                            ]
                            if len(otherWays) > 0:
                                current_way = otherWays[0]
                                block_ways.append((current_way['id'], current_way['tags']))
                                node_count = len(current_way['nodes'])
                            else:
                                break
                        else:
                            break
                        flipped = current_way['nodes'].index(current_node) == 0
                        local_index = (
                            current_way['nodes'].index(current_node)
                            if not flipped
                            else 1
                        )
                    current_node = current_way['nodes'][local_index]
                    block_nodes.append(current_node)
                    if current_node in intersection_nodes:
                        # save block
                        if block_nodes[0] in blocks:
                            similar_blocks = len([x[0] for x in blocks[block_nodes[0]] if x[0] == current_node])

                            blocks[block_nodes[0]].append(
                                (current_node, similar_blocks, {'nodes': block_nodes, 'ways': block_ways})
                            )
                        else:
                            blocks[block_nodes[0]] = [
                                (current_node, 0, {'nodes': block_nodes, 'ways': block_ways})
                            ]

                        if current_node in blocks:
                            blocks[current_node].append((block_nodes[0], 0, None))
                        else:
                            blocks[current_node] = [(block_nodes[0], 0, None)]

                        if current_node in explored_space:
                            explored_space[current_node].append(block_nodes)
                        else:
                            explored_space[current_node] = [block_nodes]
                        end_found = True


# # Print the JSON to the console in a readable way
# pp.pprint(blocks)

json.dump(blocks, open(block_output_file, 'w', encoding='utf-8'), indent=4)
