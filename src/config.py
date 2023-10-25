import math
import os
import uuid
from datetime import timedelta
from enum import Enum
from typing import Any, Literal, TypedDict
from typing_extensions import NotRequired

"----------------------------------------------------------------------------------"
"                                     File Paths                                   "
"----------------------------------------------------------------------------------"


BASE_DIR = os.path.abspath(os.path.join(__file__, "../../"))
VRP_CLI_PATH = "/home/user/.cargo/bin/vrp-cli"

CAMPAIGN_NAME = "rosselli"
AREA_BBOX = [40.5147085, -80.2215597, 40.6199697, -80.0632736]

USE_COST_METRIC = False
STYLE_COLOR = "#0F6BF5"

street_suffixes_file = os.path.join(BASE_DIR, "src", "street_suffixes.json")

region_dir = os.path.join(BASE_DIR, "regions", CAMPAIGN_NAME)
input_dir = os.path.join(region_dir, "input")

# Per-region input files
block_output_file = os.path.join(input_dir, "block_output.json")
adjacency_list_file = os.path.join(input_dir, "adjacency_list.json")
coords_node_file = os.path.join(input_dir, "coords_node.json")
overpass_file = os.path.join(input_dir, "overpass.json")
manual_match_output_file = os.path.join(input_dir, "manual_match_output.json")
mail_data_file = os.path.join(BASE_DIR, "input", "mail_data_10-20-23.json")

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

# Global input files
street_view_failed_uuids_file = os.path.join(
    BASE_DIR, "input", "street_view_failed_uuids.json"
)

turnout_predictions_file = os.path.join(
    BASE_DIR, "input", "2023_general_predictions.json"
)

VIZ_PATH = os.path.join(BASE_DIR, "viz")
# PROBLEM_PATH = os.path.join(VIZ_PATH, "problem")

"----------------------------------------------------------------------------------"
"                                     Database                                     "
"----------------------------------------------------------------------------------"
# Indices for the Redis database
VOTER_DB_IDX = 1
PLACE_DB_IDX = 2
BLOCK_DB_IDX = 3
NODE_DISTANCE_MATRIX_DB_IDX = 4
HOUSE_DISTANCE_MATRIX_DB_IDX = 5
BLOCK_DISTANCE_MATRIX_DB_IDX = 6
NODE_COORDS_DB_IDX = 7
HOUSE_IMAGES_DB_IDX = 8
STREET_SUFFIXES_DB_IDX = 9
CAMPAIGN_SUBSET_DB_IDX = 10

TESTING_DB_IDX = 11


"----------------------------------------------------------------------------------"
"                               Problem Parameters                                 "
"----------------------------------------------------------------------------------"


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


def house_value(voter_values: list[float]) -> float:
    """Calculate the value of a house based on the voters in it."""
    base_value = exponential(sum(voter_values), 1)
    return round(base_value * 100)
    # in_order = sorted(voter_values, reverse=True)
    # total = in_order[0] + (0.2 * sum(in_order[1:]))
    # return round(total)


"----------------------------------------------------------------------------------"
"                             Optimization Parameters                              "
"----------------------------------------------------------------------------------"
NUM_LISTS = 3
DEPOT = "107503392"

SEARCH_MODE_DEEP = False
TIMEOUT = timedelta(seconds=100)

Problem_Types = Enum("Problem", ["turf_split", "group_canvas", "completed_group_canvas"])
PROBLEM_TYPE = Problem_Types.group_canvas


"----------------------------------------------------------------------------------"
"                                     Constants                                    "
"----------------------------------------------------------------------------------"

GOOGLE_MAPS_API_KEY = "AIzaSyAPpRP4mPuMlyRP8YiIaEOL_YAms6TpCwM"

UUID_NAMESPACE = uuid.UUID("ccf207c6-3b15-11ee-be56-0242ac120002")

# Maximum distance between two nodes where they should be stored
ARBITRARY_LARGE_DISTANCE = 10000
MAX_TOURING_TIME = timedelta(minutes=180)
MAX_TOURING_DISTANCE = 10000
WALKING_M_PER_S = 1.2
MINS_PER_HOUSE = 1.5
CLUSTERING_CONNECTED_THRESHOLD = 100  # Meters where blocks are connected
# TODO: Reimplement Keep_apartments?
DISTANCE_TO_ROAD_MULTIPLIER = 0.5
ALD_BUFFER = 150  # Meters after a block ends where a house is still on the block
DIFFERENT_BLOCK_COST = 25

# Number of meters to store if nodes are too far away from each other
NODE_TOO_FAR_DISTANCE = 10000


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
NodeType = Enum("NodeType", ["house", "node", "other"])


def generate_pt_id(*args, **kwargs) -> str:
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
        lat = args[0]["lat"]
        lon = args[0]["lon"]
    else:
        raise ValueError("Either lat/lon or pt must be provided")
    return str("{:.7f}".format(lat)) + ":" + str("{:.7f}".format(lon))


def generate_pt_id_pair(id1: str, id2: str) -> str:
    # The middle character must never appear in an ID
    return id1 + "_" + id2


def generate_block_id_pair(id1: str, id2: str) -> str:
    # The middle character must never appear in an ID
    return id1 + "_" + id2


def generate_place_id_pair(id1: str, id2: str) -> str:
    # The middle character must never appear in an ID
    return id1 + "_" + id2


class Point(TypedDict):
    lat: float
    lon: float
    type: NodeType
    id: str


class WriteablePoint(TypedDict):
    lat: float
    lon: float


AnyPoint = Point | WriteablePoint


def to_serializable_pt(p: Point) -> WriteablePoint:
    return WriteablePoint(lat=p["lat"], lon=p["lon"])


def pt_id(p: AnyPoint) -> str:
    """
    Get the ID of a point.

    Parameters
    ----------
        p (Point): the point

    Returns
    -------
        str: the ID, if it was provided upon creation. Otherwise, an ID made up of the rounded coordinates
    """
    return (
        generate_pt_id(p["lat"], p["lon"])
        if "id" not in p.keys() or p["id"] is None
        else p["id"]
    )


house_t = dict[str, Point]
node_list_t = list[Point | WriteablePoint]


class PlaceGeography(TypedDict):
    """Geographic information about a house."""
    lat: float
    lon: float
    distance_to_start: int
    distance_to_end: int
    side: bool
    distance_to_road: int
    subsegment: tuple[int, int]


class PlaceSemantics(TypedDict):
    """Semantic information about a house."""
    display_address: str
    # For houses with multiple units, a mapping of unit numbers to a list of voter IDs
    voters: NotRequired[list[str] | dict[str, list[str]]]
    block_id: str
    city: str
    state: str
    zip: str


# TODO: rename houses to places and re-run
class Block(TypedDict):
    places: dict[str, PlaceGeography]
    nodes: node_list_t
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


class Person(TypedDict):
    name: str
    age: int
    party: Literal["D", "R", "I"]
    voter_id: str
    place: str
    place_unit: NotRequired[str]
    voting_history: dict[str, bool]
    turnout: float
    value: float


# NOTE/TODO: About to be deprecated
class HousePeople(TypedDict):
    display_address: str
    city: str
    state: str
    zip: str
    latitude: float
    longitude: float
    voter_info: list[Person]
    value: float


# NOTE/TODO: About to be deprecated
class HouseOutput(TypedDict):
    display_address: str
    city: str
    state: str
    zip: str
    uuid: str
    latitude: float
    longitude: float
    voter_info: list[Person]
    subsegment_start: int


voters_file_t = dict[str, HousePeople]


"----------------------------------------------------------------------------------"
"                                 Solution Type Hints                              "
"----------------------------------------------------------------------------------"


class Statistic(TypedDict):
    cost: float
    distance: float
    duration: float
    times: dict[str, int]


class Location(TypedDict):
    index: int


class Time(TypedDict):
    arrival: str
    departure: str


class Stop(TypedDict):
    location: Location
    time: Time
    distance: int
    load: list[int]
    activities: list[dict[str, str]]


class Tour(TypedDict):
    vehicleId: str
    typeId: str
    shiftIndex: int
    stops: list[Stop]


class Solution(TypedDict):
    statistic: Statistic
    tours: list[Tour]
    unassigned: list[Any]


"----------------------------------------------------------------------------------"
"                                  Problem Type Hints                              "
"----------------------------------------------------------------------------------"


class Profile(TypedDict):
    matrix: str


class Costs(TypedDict):
    fixed: int
    distance: int
    time: int


class ShiftStart(TypedDict):
    earliest: str
    location: Location


class ShiftEnd(TypedDict):
    latest: str
    location: Location


class Shift(TypedDict):
    start: ShiftStart
    end: ShiftEnd


class Limit(TypedDict):
    max: int
    start: str
    end: str


class Dispatch(TypedDict):
    location: Location
    limits: list[Limit]


class ShiftDispatch(TypedDict):
    start: ShiftStart
    dispatch: list[Dispatch]


class VehicleLimits(TypedDict):
    shiftTime: int
    maxDistance: int


class Vehicle(TypedDict):
    typeId: str
    vehicleIds: list[str]
    profile: Profile
    costs: Costs
    shifts: list[Shift]
    capacity: list[int]
    limits: VehicleLimits


class VehicleProfile(TypedDict):
    name: str


class Fleet(TypedDict):
    vehicles: list[Vehicle]
    profiles: list[VehicleProfile]


class PlaceTW(TypedDict):
    location: Location
    duration: int
    times: list[list[str]]


class Place(TypedDict):
    location: Location
    duration: int


class Service(TypedDict):
    places: list[PlaceTW | Place]


class Job(TypedDict):
    id: str
    services: list[Service]
    value: int


class Plan(TypedDict):
    jobs: list[Job]


class Objective(TypedDict):
    type: str


class Problem(TypedDict):
    plan: Plan
    fleet: Fleet
    objectives: list[list[Objective]]


class DistanceMatrix(TypedDict):
    profile: str
    travelTimes: list[int]
    distances: list[int]


"----------------------------------------------------------------------------------"
"                             Optimization Parameters                              "
"----------------------------------------------------------------------------------"
OPTIM_COSTS = Costs(fixed=0, distance=3, time=1)

OPTIM_OBJECTIVES = [
    [Objective(type="maximize-value")],
    [Objective(type="minimize-cost")],
]

# OPTIM_OBJECTIVES = [
#     [Objective(type="maximize-value")],
# ]
