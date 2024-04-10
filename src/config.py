import math
import os
import uuid
from datetime import timedelta
from enum import Enum
from typing import Literal, TypedDict
from typing_extensions import NotRequired
import json

BASE_DIR = os.path.abspath(os.path.join(__file__, "../../"))

"----------------------------------------------------------------------------------"
"                             Problem Parameters                                   "
"----------------------------------------------------------------------------------"

problem_path = os.path.join(BASE_DIR, "problem.json")
problem_params = json.load(open(problem_path))

assert "campaign_id" in problem_params
CAMPAIGN_ID = problem_params["campaign_id"]

Problem_Types = Enum("Problem", ["turf_split", "group_canvas"])

PROBLEM_TYPE = Problem_Types.group_canvas if "depot" in problem_params else Problem_Types.turf_split

assert "num_routes" in problem_params and int(problem_params["num_routes"])
NUM_ROUTES = problem_params["num_routes"]

if PROBLEM_TYPE == Problem_Types.turf_split:
    assert "depot" not in problem_params, \
        "For turf_split, do not provide any depots. They are decided by optimality at runtime"
    DEPOT = None
elif PROBLEM_TYPE == Problem_Types.group_canvas:
    assert "depot" in problem_params and isinstance(problem_params["depot"], str), \
        "For group_canvas, you must provide num_lists and a single depot location"
    DEPOT = problem_params["depot"]

assert "timeout_s" in problem_params and int(problem_params["timeout_s"])
TIMEOUT = timedelta(seconds=problem_params["timeout_s"])

# NOTE: Right now, there's no standardized way to map campaign id to a physical area
# This may not need to be solved, if we end up processing entire states and such in advance.
AREA_BBOX = [40.5147085, -80.2215597, 40.6199697, -80.0632736]

"----------------------------------------------------------------------------------"
"                                     File Paths                                   "
"----------------------------------------------------------------------------------"
street_suffixes_file = os.path.join(BASE_DIR, "src", "street_suffixes.json")

region_dir = os.path.join(BASE_DIR, "regions", CAMPAIGN_ID)
input_dir = os.path.join(region_dir, "input")

# Per-region input files
block_output_file = os.path.join(input_dir, "block_output.json")
adjacency_list_file = os.path.join(input_dir, "adjacency_list.json")
coords_node_file = os.path.join(input_dir, "coords_node.json")
overpass_file = os.path.join(input_dir, "overpass.json")
manual_match_output_file = os.path.join(input_dir, "manual_match_output.json")
mail_data_file = os.path.join(BASE_DIR, "input", "mail_data_10-27-23.json")

# Map addresses to block IDs
addresses_file = os.path.join(region_dir, "addresses.json")
manual_match_input_file = os.path.join(region_dir, "manual_match_input.json")
reverse_geocode_file = os.path.join(region_dir, "reverse_geocode.json")

id_to_addresses_file = os.path.join(region_dir, "id_to_addresses.json")

# Per-problem pickle files
optimizer_points_pickle_file = os.path.join(region_dir, "points.pkl")
clustering_pickle_file = os.path.join(region_dir, "clustering.pkl")

# Default problem files, if not given per-problem
address_pts_file = os.path.join(BASE_DIR, "input", "address_pts.csv")
default_problem_path = os.path.join(BASE_DIR, "optimize", "problem.json")
default_solution_path = os.path.join(BASE_DIR, "optimize", "solution.json")
default_distances_path = os.path.join(BASE_DIR, "optimize", "distances.json")

details_file = os.path.join(region_dir, "details.json")
files_dir = os.path.join(region_dir, "files")
vizs_dir = os.path.join(region_dir, "vizs")

# Global input files
street_view_failed_uuids_file = os.path.join(
    BASE_DIR, "input", "street_view_failed_uuids.json"
)

turnout_predictions_file = os.path.join(
    BASE_DIR, "input", "2023_general_predictions.json"
)

VIZ_PATH = os.path.join(BASE_DIR, "viz")

"----------------------------------------------------------------------------------"
"                                     Database                                     "
"----------------------------------------------------------------------------------"
# Indices for the Redis database
VOTER_DB_IDX = 1
ABODE_DB_IDX = 2
BLOCK_DB_IDX = 3
NODE_DISTANCE_MATRIX_DB_IDX = 4
BLOCK_DISTANCE_MATRIX_DB_IDX = 6
NODE_COORDS_DB_IDX = 7
ABODE_IMAGES_DB_IDX = 8
STREET_SUFFIXES_DB_IDX = 9
CAMPAIGN_SUBSET_DB_IDX = 10

TESTING_DB_IDX = 11


"----------------------------------------------------------------------------------"
"                               Problem Parameters                                 "
"----------------------------------------------------------------------------------"

USE_COST_METRIC = False
MAX_TOURING_TIME = timedelta(minutes=180)
TIME_AT_ABODE = timedelta(minutes=1.5)
MAX_TOURING_DISTANCE = 6000
WALKING_M_PER_S = 1.2
SUPER_CLUSTER_NUM_ABODES = 500


def sigmoid(x: float, k: float, a: float) -> float:
    """Sigmoid function with steepness parameter k and shift parameter a.

    This should be used on voter probabilities when the metric needs to be exaggerated,
    so low probabilities will map even lower and high probabilities even higher.
    """
    return 1 / (1 + math.exp(-k * (x - a)))


def exponential(x: float, k: float) -> float:
    """Exponential function with steepness parameter k, scaled to [0, 1].

    This should be used on voter probabilities when the metric needs to be diminished,
    so low probabilities will map higher.
    """
    return (1 - math.exp(-k * x)) / (1 - math.exp(-k))


def voter_value(party: Literal["D", "R", "I"], turnout: float) -> float:
    # We use the normalized sigmoid function to exaggerate the differences between turnout probabilities.
    # Here, more steepness (k) means more exaggeration, and the (a) value is at what probability the function
    # is ambivalent (lower means fewer low-propensity voters are targeted)
    # base_value = sigmoid(turnout, 10, 0.4)
    # normalized = (base_value - sigmoid(0, 10, 0.4)) / (
    #     sigmoid(1, 10, 0.4) - sigmoid(0, 10, 0.4))
    base_value = exponential(turnout, 5)

    if turnout < 0.1:
        base_value = 0

    return base_value if party in ["D", "I"] else 0


def abode_value(voter_values: list[float]) -> float:
    """Calculate the value of a abode based on the voters in it."""
    base_value = exponential(sum(voter_values), 1)
    return round(base_value * 100)
    # in_order = sorted(voter_values, reverse=True)
    # total = in_order[0] + (0.2 * sum(in_order[1:]))
    # return round(total)


# NOTE: May be deprecated soon as distance matrices are now stored per-problem to save memory
class Singleton(type):
    _instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instance


"----------------------------------------------------------------------------------"
"                                     Constants                                    "
"----------------------------------------------------------------------------------"
STYLE_COLOR = "#0F6BF5"

GOOGLE_MAPS_API_KEY = os.get_env("GOOGLE_MAPS_API_KEY")

UUID_NAMESPACE = uuid.UUID("ccf207c6-3b15-11ee-be56-0242ac120002")

# Distance between points (nodes, blocks, abodes, etc.) beyond which they are not stored
MAX_STORAGE_DISTANCE = 1600

# The return value for distance functions when the distance is not stored
ARBITRARY_LARGE_DISTANCE = 10000

# How much the distance to the road should affect the distance between abodes
# (inaccuracy and imperfect perception of distance make this generally less than 1)
DISTANCE_TO_ROAD_MULTIPLIER = 0.5

# Number of meters after a block ends where an abode is still considered to be on the block
ALD_BUFFER = 150

# The cost of crossing the street (technically, in meters)
DIFFERENT_BLOCK_COST = 25

TERMINAL_WIDTH = os.get_terminal_size().columns


# Cost of crossing the street (technically, in meters)
DIFFERENT_SIDE_COST = {
    "motorway": 400,
    "trunk": 400,
    "primary": 100,
    "secondary": 60,
    "tertiary": 35,
    "unclassified": 20,
    "residential": 20,
    "service": 10,
}

# Cost of crossing the street (technically, in meters)
ROAD_WIDTH = {
    "motorway": 20,
    "trunk": 16,
    "primary": 12,
    "secondary": 10,
    "tertiary": 8,
    "unclassified": 8,
    "residential": 6,
    "service": 0,
}


"----------------------------------------------------------------------------------"
"                                       Type Hints                                 "
"----------------------------------------------------------------------------------"
# NOTE: the 'type' and 'id' attributes are not required. When using Python 3.11,
# wrap these attributes in the 'typing.NotRequired' hint to eliminate errors on instance creation.
# On earlier versions, either suppress or ignore these errors: they do not affect json export or reading.

# Define the Nodetypes as an enum
NodeType = Enum("NodeType", ["abode", "node", "other"])


# region Point-like object type hints
# NOTE: We stored two different kinds of points, one for internal use and one for export.
# InternalPoint allows for passing around points which are associated with ids and types directly,
# instead of needing to pass around metadata separately.
PT_ID_SEPARATION = ":"
PT_ID_PRECISION = 7
ID_PAIR_SEPARATION = "_"


class InternalPoint(TypedDict):
    lat: float
    lon: float
    type: NodeType
    id: str


class WriteablePoint(TypedDict):
    lat: float
    lon: float


Point = InternalPoint | WriteablePoint


def generate_id_pair(id1: str, id2: str) -> str:
    """
    Generate a unique ID for a pair of points, given their IDs.

    Parameters
    ----------
        id1 (str): the ID of the first point
        id2 (str): the ID of the second point

    Returns
    -------
        str: the ID
    """
    return id1 + ID_PAIR_SEPARATION + id2


def pt_id(p: Point) -> str:
    """
    Get the ID of a point, or generate one if it does not have one.

    Parameters
    ----------
        p (Point): the point

    Returns
    -------
        str: the ID, if it was provided upon creation. Otherwise, an ID made up of the rounded coordinates
    """
    def generate_pt_id(*args, **kwargs) -> str:
        """
        Generate a unique ID for a point, given some representation of a point as input.

        Returns
        -------
            str: the ID

        Examples
        --------
            >>> generate_pt_id(lat=40.5, lon=-80.2)
            '40.5000000:-80.2000000'
            >>> generate_pt_id(pt={"lat": 40.5, "lon": -80.2})
            '40.5000000:-80.2000000'
            >>> generate_pt_id({"latitude": 40.5, "longitude": -80.2})
            '40.5000000:-80.2000000'
            >>> generate_pt_id(40.5, -80.2)
            '40.5000000:-80.2000000'
        """
        if "lat" in kwargs and "lon" in kwargs:
            lat = kwargs["lat"]
            lon = kwargs["lon"]
        elif "pt" in kwargs:
            lat = kwargs["pt"]["lat"]
            lon = kwargs["pt"]["lon"]
        elif len(args) == 2:
            lat = args[0]
            lon = args[1]
        elif len(args) == 1:
            if "lat" in args[0] and "lon" in args[0]:
                lat = args[0]["lat"]
                lon = args[0]["lon"]
            elif "latitude" in args[0] and "longitude" in args[0]:
                lat = args[0]["latitude"]
                lon = args[0]["longitude"]
            else:
                raise ValueError("Either lat/lon or pt must be provided")
        else:
            raise ValueError("Either lat/lon or pt must be provided")

        return str(round(lat, PT_ID_PRECISION)) + PT_ID_SEPARATION + str(round(lon, PT_ID_PRECISION))

    if "id" not in p.keys() or p["id"] is None:
        return generate_pt_id(p["lat"], p["lon"])

    return p["id"]
# endregion


class AbodeGeography(TypedDict):
    """
    An abode's geographic information, relative to the block it is on.

    Notes
    -----
    Block directionality in storage matters, since start and end rely on consistent direction.
    """
    id: str
    point: WriteablePoint
    distance_to_start: int
    distance_to_end: int
    side: bool
    distance_to_road: int
    subsegment_start: int
    subsegment_end: int


class Abode(TypedDict):
    """An abode's semantic information"""
    id: str
    display_address: str
    # For abodes with multiple units, a mapping of unit numbers to a list of voter IDs
    voter_ids: NotRequired[list[str] | dict[str, list[str]]]
    block_id: str
    city: str
    state: str
    zip: str


class SubAbode(TypedDict):
    """A part of an abode, representing the abode's semantic and geographic information, and containing a subset of all the voters living at this abode."""
    abode_id: str
    point: WriteablePoint
    distance_to_start: int
    distance_to_end: int
    side: bool
    distance_to_road: int
    subsegment_start: int
    subsegment_end: int
    display_address: str
    voter_ids: NotRequired[list[str] | dict[str, list[str]]]
    block_id: str
    city: str
    state: str
    zip: str


class Block(TypedDict):
    """
    A segment of a street, containing abodes.

    Notes
    -----
    Block directionality matters (the nodes list cannot be reversed), since abodes store directional information.
    """
    id: str
    abodes: dict[str, AbodeGeography]
    nodes: list[Point]
    type: str


blocks_file_t = dict[str, Block]


"----------------------------------------------------------------------------------"
"                               Output File Type Hints                             "
"----------------------------------------------------------------------------------"

tracked_elections = Enum(
    "elections",
    [
        "primary_2023",
        "general_2022",
        "primary_2022",
        "general_2021",
        "primary_2021",
        "general_2020",
        "primary_2020",
        "general_2019",
        "primary_2019",
    ],
)


# Mapping from voter file column names to elections
# This is retreived directly from the "Election Map" file from the county
voter_file_mapping = {
    7: tracked_elections.primary_2019,
    8: tracked_elections.general_2019,
    9: tracked_elections.primary_2020,
    10: tracked_elections.general_2020,
    21: tracked_elections.primary_2021,
    22: tracked_elections.general_2021,
    25: tracked_elections.primary_2022,
    26: tracked_elections.general_2022,
    30: tracked_elections.primary_2023,
}


class Voter(TypedDict):
    id: str
    name: str
    age: int
    party: Literal["D", "R", "I"]
    abode_id: str
    abode_unit: NotRequired[str]

    # Mapping from election date to whether the voter voted
    voting_history: dict[str, bool]
    turnout: float


# NOTE/TODO: About to be deprecated
class AbodeOutput(TypedDict):
    display_address: str
    city: str
    state: str
    zip: str
    uuid: str
    latitude: float
    longitude: float
    voter_info: list[Voter]
    subsegment_start: int
