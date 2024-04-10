# Edna Backend

This package is the backend for the Edna canvassing technology, and temporarily contains the processing for ingesting and pre-processing campaign data.

![Backend](https://github.com/aaronzberger/EdnaBackend/assets/35245591/4d29edb2-2e09-4cbc-9e32-e68d179aadd3)

![App](https://github.com/aaronzberger/EdnaBackend/assets/35245591/9abe668d-5a0b-4d4c-bf3a-bfa3da9ae214)

![Dashboard](https://github.com/aaronzberger/EdnaBackend/assets/35245591/7499f66e-9c09-4b55-b4c3-b388a75042f8)

## Table of Contents
- [Overview](#Overview)
- [Setup](#Setup)
- [Details](#Details)
  - [Data Structuring](#Data-Structuring)
    - [Typing](#Typing)
    - [Distance matrices](#Distance-matrices)
  - [Pre-processing](#Pre-processing)
    - [Geographic data fetching](#Geographic-data-fetching)
    - [Geographic data parsing](#Geographic-data-parsing)
    - [Block creation](#Block-creation)
    - [Universe matching](#Universe-matching)
  - [Route Optimization](#Route-Optimization)
    - [Problem Formulation](#Problem-Formulation)
    - [Solving](#Solving)
    - [Post-processing](#Post-processing)

  
# Overview
Welcome to the Edna canvassing and campaign management backend! This package contains the code for the backend, which is responsible for:
 - Ingesting campaign data
 - Pre-processing the data into a format which is usable by the rest of the system
 - Optimizing routes for canvassing
 - Post-processing the routes into a format which is usable by the rest of the system


# Setup
This project uses Docker to manage all dependencies, so the setup is simple.
1. Install Docker

All the Docker components (routing, database, backend) communicate via a Docker network, so we need to create one.

2. Run `docker network create edna`.

Next, we need to setup [OSRM](https://project-osrm.org/) (the routing engine), which runs locally for speed.

3. In the route directory, `mkdir osrm`. This will be the directory which contains the OSRM data.
4. Run `../docker/run_osrm.sh`. This will download the OSRM data (for Pennsylvania), process it, and run the OSRM server. This may take a while, and will use a bit less than 3GB of space.

Next, we need to setup and run the Redis server.

5. Run `./docker/run_db.sh`, which will start the Redis server (if this doesn't work, simply copy the command in the script and run it manually). If an error occurs, delete the old container and try again.

Finally, we can run the backend.

6. Run `./docker/build_backend.sh`, which will build the backend Docker image. This takes about 3 minutes.
7. Run `./docker/run_backend.sh` from the root directory, which will run the backend Docker image. This will mount the base directory, and all changes made within the container will be reflected on the host machine (and vice versa). However, Git may not work within the container (unless you'd like to set it up manually), so simply use Git on the host machine.


# Details
The usage of this package is documented throughout the sections below, instead of in a separate section, due to the number of scripts and steps involved. First, let's explore the general principles we use to structure the data.

There are parts of the code which are not fully documented below, as they are used only by sub-components described below.

## Data Structuring
There are multiple de-coupling steps which make Edna adaptable, scalable, and understandable. Within the backend, de-coupling the structure of the data with the functionality of the backend is critical to being able to adapt to new functionality and data.

The schema we use is documented at the bottom of the Swagger API documentation [here](https://aaronzberger.github.io/EdnaArchitecture/). The rest of the API documentation is also worth exploring, to better understand the purpose of the backend (the outputs).

The geographic structuring is as follows: a `Block` is the segment of road between two possible turning points. A `Block` contains `node`s, which are the `Point`s making up the grographical structure of the line segment. A `Block` also contains `Abode`s, which are places where voters live (this could be a house, apartment within a building, etc.). `Abode`s contain one or more `Voter`s.

### Typing
The objects described in this schema are implemented as `TypedDict` objects in Python, with nearly precisely the same attributes as are on the Swagger documentation (and we attempt to keep these consistent). For example, look at the `Voter` type in the Swagger schema, and compare it with the `Voter` class in `src/config.py` (they're the same). We use `TypedDict` objects for easy reading and writing to JSON, and for easy stringification for storing in Redis.

### Distance Matrices
Throughout the backend, it becomes necessary to frequently calculate distances between `Block`s, `node`s, and `Abode`s. Due to the large number of `Abode`s, we choose to pre-process the distance matrices for `node`s and `Block`s (and store them in Redis, see TODO below, which adds these distances if needed for each new campaign), but not for `Abode`s. For `Abode`s, we calculate the distance matrices only at the time of optimization, in small clusters. Thus, there is no central storage of `Abode` distances.

The `node` distance matrix (in `src/distances/nodes.py`) is the only distance matrix which is generated by actually performing routing. It stores the distances between each intersection.

The `Block` distance matrix (which stores the shortest distance between any two endpoints between two `Block`s) simply uses the `node` distance matrix to calculate the distance between each `Block` (in `src/distances/blocks.py`).

The `Abode` distance matrix similarly uses the `Block` distance matrix, along with the information within each `AbodeGeography` object to calculate distances (in `src/distances/abodes.py`).


## Pre-processing
The goal of the pre-processing is to take a campaign's geographic information (what areas they are running in), and prepare the data for canvassing. The inputs to the pre-processing are:
 - The campaign's "universe" file, which contains a subset of the public voter file, containing only the voters the campaign wants to target (see `regions/dev_1/input/universe.csv` for an example).
 - Geocoding information, which maps an address to coordinates. Right now, we test in Alleghney County, PA, which releases a file with this information (see `input/address_pts.csv`).

### Geographic Data Fetching
First, we use OpenStreetMap (OSM) to retrieve a geographic understanding of the region requested: blocks and nodes. Run `src/pre_processing/get_data.py` to query OSM for the bounding box defined in `src/config.py`. This may take a while, but finally generates a large file called `overpass.json`.

### Geographic Data Parsing
Next, we process the raw data in `overpass.json` into a more usable format. Run `src/pre_processing/preprocess_data.py` to generate `block_output.json`, which contains all the blocks and nodes in the region. This file is structured as follows:
```
{
    "node_id": [
        [
            "another_node_id",
            0 or 1,
            {
                "nodes": [list of node IDs],
                "ways": [
                  [
                    way ID
                    way info
                  ],
                  [
                    another way ID
                    another way info
                  ],
                  ...
                ]
            }
        ],
        ...
    ],
    ...
}
```

We also can now store the coordinates of all the nodes in the region for quick access later. Run `src/pre_processing/parse_json.py` to store the nodes from `overpass.json` file into Redis.

### Block Creation
Now, with our understanding of nodes and blocks, we can create the actual `Block` objects and store them in Redis. This involves associating the abodes with the blocks, and creating the `Abode` and `Block` objects fully (apart from adding voters to the `Abode`s). Run `src/pre_processing/make_blocks.py`.

### Universe Matching
Finally, we can associate the `universe.csv` file with these `Abode`s and add the voters to their respective `Abode`s. Run `src/pre_processing/process_universe.py` to do this. This also pre-populates the distance matrices for `node`s and `Block`s to save time later.


## Route Optimization
Now that we have a geographic and semantic understanding of a campaign's data, it's time to simply do some route optimization to find the best canvassing routes. First, let's explore what a campaign might actually want:

### Problem Formulation
The first type of problem is called the *Group Canvas*. This occurs when a campaign wants to host an event at some place, and have some number of volunteers complete canvassing routes from this point.

The second type of problem is called the *Turf Split*, which is the bulk of the novel functionality. This occurs at the beginning of a campaign, when all the routes are generated, and periodically, as the number of projected houses the campaign will visit throughout the campaign changes, and as group canvasses are separately performed.

Both formulations reduce to the same problem, and are both solved using the `BaseSolver` in `src/optimize/base_solver.py`. This solver simply calls [OR-Tools](https://developers.google.com/optimization/routing/vrp) (the best open-source VRP solver) with some starting locations, number of routes, and other route parameters.

#### Group Canvas
For the *Group Canvas* problem (in `src/optimize/group_canvas.py`), the reduction is simple: given the voter universe, a starting location, and the number of routes (along with other route parameters), it finds the probable area around the starting location needed (to reduce the problem complexity, instead of running on the entire campaign's area), and then generates these routes.

**In the future, we may wish to add functionality to account for driving to a starting location. Thus, this may turn into a turf split problem with a central location.**

#### Turf Split
This reduction is a bit more complicated. In this problem, we have a huge (usually the entire campaign's) area, and wish to choose the starting locations (among all intersections in the area) which maximizes the number of houses hit in the least time.

**In the future, we hope to use optimization to determine the optimal set of depots to use**. For now, we use the following heuristic, which we call *Super Clustering*: we cluster the large area into manageable chunks (about 500 `Abode`s each), then find depots within each cluster (by clustering again and using the centroids), and then run the base solver with these depots. This is implemented in `src/optimize/turf_split.py`.

### Solving
Now that we have a problem formulation, we can solve it. This is fully done for us by OR-Tools. The problem setup for OR-Tools and calling is done in `src/optimize/base_solver.py`.

### Post-processing
Since the output of the VRP is a list of points for each route, we need to convert this into our desired format. The reason we need `Block`s and cannot simply display raw points to users is the following:
 - Canvassers and campaigns have an easier time understanding larger structures (blocks) than individual points, and giving a route (as Google Maps does) is much easier to follow than giving individual points.
 - We'd like the ability to impose some "walkability" heuristics into the routes, like not backtracking on yourself (where the optimizer may be indifferent but a human could get confused).
 - **In the future, we'd also like to give to the user the side of the street they should be walking on, just like Google Maps walking directions does.**

The long post-processing script in `src/optimize/post_process.py` does the following performs this re-association and imposes the heuristics needed. It also displays visualizations of the routes, and produces files which will be used by the dashboard and app to visualize and walk the routes.
