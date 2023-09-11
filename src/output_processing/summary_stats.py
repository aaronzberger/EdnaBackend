# Read in the details file and the output files and add some additional information

import os
import sys
import json
from typing import Any
import numpy as np


# Read in the details file
details_file = sys.argv[1]
details: dict[str, Any] = json.load(open(details_file))

output_dir = sys.argv[2]

distances = []
num_houses = []
num_voters = []
num_routes = 0
times = []


for list_id in details.keys():
    # Read in the output file
    list_file = os.path.join(output_dir, f"{list_id}.json")

    num_routes += 1
    num_voters.append(details[list_id]["num_voters"])
    num_houses.append(details[list_id]["num_houses"])
    distances.append(details[list_id]["distance"])

    # The time is distance * walking_speed + time_per_house * num_houses
    times.append(details[list_id]["distance"] * 1.2 + 1.5 * details[list_id]["num_houses"])


print(f"Mean houses per route = {np.mean(num_houses)}, median = {np.median(num_houses)}")
print(f"Mean voters per route = {np.mean(num_voters)}, median = {np.median(num_voters)}")
print(f"Mean distance per route = {np.mean(distances)}, median = {np.median(distances)}")
print(f"Mean time per route = {np.mean(times)}, median = {np.median(times)}")
print(f"Number of routes = {num_routes}")
print(f"Total houses {sum(num_houses)}, total voters {sum(num_voters)}, total distance {sum(distances)}")
print(f"Total time {sum(times)}")
