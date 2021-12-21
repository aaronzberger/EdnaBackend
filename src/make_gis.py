import csv
import json
import pickle

file = json.load(open('/home/aaron/Downloads/blockoutput.json', 'r'))

coords_file = json.load(open('/home/aaron/Downloads/groupednodes.json', 'r'))

with open('/home/aaron/walk_list_creator/output.csv', 'w') as write:
    writer = csv.DictWriter(write, fieldnames=['ID', 'BlockID', 'Latitude', 'Longitude'])
    for start_node_list in file:
        for block in file[start_node_list]:
            if block[1] is not None:
                for node in block[1]['nodes']:
                    try:
                        writer.writerow({
                            'ID': node,
                            'BlockID': '{}{}'.format(start_node_list, block[0]),
                            'Latitude': coords_file[str(node)][0]['lat'],
                            'Longitude': coords_file[str(node)][0]['lon']
                        })
                    except KeyError:
                        pass