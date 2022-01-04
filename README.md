# WLC Pre-processing

This pipeline generates the `associated.csv` file and visualizations for it.

<img src="https://user-images.githubusercontent.com/35245591/147997445-bb23eec5-8b0c-480c-b283-603f23c5d218.png" alt="Associations" width="589" height="412">

## Table of Contents
- [Usage](#Usage)
- [Retrieve Data](#Retrieve-Data)
  - [Block & Node Info](#Retrieve-Data)
  - [House Coordinates](#House-Coordinates)
- [Association](#Association)
- [Visualization](#Pre-processing)
  - [Pre-processing](#Pre-processing)
  - [Running](#GIS-Visualizations)


## Usage

Change the `BASE_DIR` variable in `utils.py` to the path of your `WLC-Preprocessing` folder.

 - Run `get_data.py` to generate an `input/area.json` file, containing all nodes and ways in the desired area
 - Run `parse_json.py` to generate the hash table with all the nodes, saved to a pickle
 - Run `preprocess_data.py` to generate `input/block_output.json`, containing blocks, their nodes, and their attributes
 - Run `associate_houses.py` to generate `associated.csv`, which associates each house with a block

 For visualization:
 - Run `make_gis.py` to generate `qgis/blocks.csv`, which contains all nodes in all ways, identified by block ID.
 - Open `visualizations.qgz` to run the visualization. Try to fix any files that are not found by rearranging file locations.


## Retrieve Data

`preprocess_data.py` generates `block_output.json`, which contains all nodes and blocks in a specified area. Each element in the outer dict is a list of blocks, defined as follows:

[//]: # (Apologies for all these tabs)

"Node 1 ID"<br>
&emsp;&emsp;[<br>
&emsp;&emsp;&emsp;&emsp;[<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Node 2 ID (int),<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;0 or 1 (find out what this means),<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;way_details dict or null {<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;"nodes" (list of nodes in this block)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;"ways" (list of lists, each with a way ID and the way info)<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;}<br>
&emsp;&emsp;&emsp;&emsp;]<br>
&emsp;&emsp;&emsp;&emsp;[<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Node 2 ID<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;0 or 1<br>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;way_details dict or null<br>
&emsp;&emsp;&emsp;&emsp;]<br>
&emsp;&emsp;&emsp;&emsp;...<br>
&emsp;&emsp;]<br>

Since a block is defined by its two endpoints and every node is guaranteed an element in this file, there are two instances for each block. The first contains a full element and the second has the way_details dict replaced with null, indicating it is a duplicate.

To find a block, choose one node endpoint and find the block list in the file. Iterate through the blocks and find the one where the Node 2 ID matches the desired ID. If the way_details dict is null, search for the block list using the Node 2 ID.


## House Coordinates

`address_pts.csv` contains the coordinates of all buildings in Allegheny County. It is provided by the county through [this](#https://data.wprdc.org/dataset/allegheny-county-addressing-address-points2) dataset. However, in November 2021, it was updated and the latitude and longitude fields were emptied. The file here was retrieved before this change and contains the coordinates, along with some other useful info about each building, like street address.


## Association

`associate_houses.py` generates `associated.csv`. It iterates through each house in the `address_pts.csv` file and looks for matching street names in block_output.json. For each matching name, it checks the distance of the house from each segment of the way. The closest distance is the way that the house belongs to.

It saves the latitude, longitude, and block ID of each association via `associated.csv`. The block ID is the concatenation of the start node ID, end node ID, and block ID (9 + 9 + 1 = 19 characters).


## Pre-processing

`make_gis.py` generates `qgis/blocks.csv`. For the purpose of visualization, it is useful to have, for each node on a street, its node ID, block ID, and coordinates together. This way, we can group these in QGIS via their block ID.

This script uses `block_output.json` and `hash_nodes.pkl` and simply concatenates the information to the final csv by searching the hash table for each node ID to fetch the coordinates.


## GIS Visualizations

To get the visualizations, simply open visualizations.gqz with QGIS Desktop. The instructions below are for re-creating this visualization from scratch:

 - Add OSM Standard Layer (copy from another project or go to Layer -> Add Layer -> Add Vector Layer -> XYZ). This adds the map.
 - Add `qgis/blocks.csv` layer (Layer -> Add Layer -> Add Delimited Text Layer)
 - Connect these points to make the blocks via the `BlockID` field (Processing -> Toolbox -> Points to Path. Set 'Input point layer' to the `blocks` layer, set 'Order field' AND 'Group field' to `BlockID`). Edit the style of this layer as desired to make the lines more visible. Make this layer permanent (*Right click layer* -> Make Permanent -> *Save as GeoJSON*)
 - Add houses and their block IDs, via `associated.csv` (Layer -> Add Layer -> Add Delimited Text Layer. *Note: Go to 'Record and Field Options' and ensure 'Detect field types' is unchecked, or else block_id will be rounded per float imprecision*)
 - Create a new virtual layer to define colors for each block:
    - Create a new layer with the attribute table as all unique block IDs (Layer -> Create Layer -> New Virtual Layer. In 'Query', type 'SELECT DISTINCT BlockID FROM blocks')
    - Create an R, G, and B field (Open Attribute Table -> Open Field Calculator. Use '3' for field length and 'R', 'G', and then 'B' for field name. For expression, type 'rand(10, 255)')
    - This layer will unfortunately re-randomize every time you zoom, so: Export -> Save Features As... -> GeoJSON with name 'perm_colors'
    - To apply the colors to the blocks, copy this into Properties... ->  Simple Line -> Color Equation Edit...
    ```
    color_rgb(
        attribute(get_feature('perm_colors', 'BlockID',  BlockID), 'R'),
        attribute(get_feature('perm_colors', 'BlockID',  BlockID), 'G'),
        attribute(get_feature('perm_colors', 'BlockID',  BlockID), 'B'))
    ```
    - To apply to houses, use the same expression.