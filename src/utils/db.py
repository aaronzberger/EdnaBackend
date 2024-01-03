"""
Temporary DB Documentation
--------------------------

Node Coords DB (NODE_COORDS_DB_IDX)
    Key: Node ID (from OSM)
    Value: Dict with keys 'lat' and 'lon'

Blocks (BLOCK_DB_IDX):
    Key: Block ID (from pre-processing, the concatenation of end node IDs and an identifier)
    Value: Block object (houses, nodes, type)

Places (PLACE_DB_IDX):
    Key: Place ID (uuid)
    Value: PlaceSemantics object

Voters (VOTER_DB_IDX):
    Key: Voter ID
    Value: Person object

Node Distances (NODE_DISTANCE_MATRIX_DB_IDX):
    Key: str representing both nodes, created from config.generate_pt_id_pair
    Value: str castable to float representing distance between nodes

Place Distances (PLACE_DISTANCE_MATRIX_DB_IDX):
    Key: str representing both places, created from config.generate_place_id_pair
    Value: 0-8 digit number: first four digits representing distance, last four digits representing some cost

"""

import json

import redis
from src.config import Singleton


class Database(metaclass=Singleton):
    def __init__(self):
        self.db = redis.Redis(
            host="redis-container",
            port=6379,
            password="edna12",
            decode_responses=True,
        )

        if not self.db.ping():
            raise ConnectionError("Redis server failed to respond to ping.")

    def set_dict(self, key: str, value: dict, database: int):
        self.set_str(key, json.dumps(value), database)

    def get_dict(self, key: str, database: int) -> dict:
        retrieved = self.get_str(key, database)
        return json.loads(retrieved)

    def set_str(self, key: str, value: str, database: int):
        self.db.select(database)
        self.db.set(key, value)

    def set_multiple_str(self, key_value_pairs: dict[str, str], database: int):
        self.db.select(database)
        self.db.mset(key_value_pairs)

    def get_str(self, key: str, database: int) -> str:
        self.db.select(database)
        return self.db.get(key)  # type: ignore

    def get_multiple_str(self, keys: list[str], database: int) -> dict[str, str]:
        self.db.select(database)
        values: list[str] = self.db.mget(keys)  # type: ignore
        return dict(zip(keys, values))

    def get_all_str(self, database: int) -> dict[str, str]:
        return self.get_multiple_str(self.get_keys(database), database)  # type: ignore

    def add_to_list(self, key: str, value: str, database: int):
        self.db.select(database)
        self.db.lpush(key, value)

    def get_list(self, key: str, database: int):
        self.db.select(database)
        return self.db.lrange(key, 0, -1)

    def get_list_length(self, key: str, database: int) -> int:
        self.db.select(database)
        return self.db.llen(key)  # type: ignore

    def add_to_set(self, key: str, value: str, database: int):
        self.db.select(database)
        self.db.sadd(key, value)

    def get_set(self, key: str, database: int) -> set[str]:
        self.db.select(database)
        return self.db.smembers(key)  # type: ignore

    def get_set_length(self, key: str, database: int) -> int:
        self.db.select(database)
        return self.db.scard(key)  # type: ignore

    def is_in_set(self, key: str, value: str, database: int) -> bool:
        self.db.select(database)
        return self.db.sismember(key, value)  # type: ignore

    def exists(self, key: str, database: int) -> bool:
        self.db.select(database)
        return self.db.exists(key) > 0  # type: ignore

    def delete(self, key: str, database: int):
        self.db.select(database)
        self.db.delete(key)

    def get_keys(self, database: int) -> list[str]:
        self.db.select(database)
        return self.db.keys()  # type: ignore

    def get_multiple_dict(self, keys: list[str], database: int) -> dict[str, dict]:
        self.db.select(database)
        values: list[str] = self.db.mget(keys)  # type: ignore
        return dict(zip(keys, [None if v is None else json.loads(v) for v in values]))  # type: ignore

    def get_all_dict(self, database: int) -> dict[str, dict]:
        return self.get_multiple_dict(self.get_keys(database), database)

    def set_multiple_dict(self, key_value_pairs: dict[str, dict], database: int):
        self.db.select(database)
        casted: dict[str, str] = {
            str(k): json.dumps(v) for k, v in key_value_pairs.items()
        }
        self.db.mset(casted)

    def get_type(self, key: str, database: int):
        self.db.select(database)
        return self.db.type(key)

    def clear_db(self, database: int):
        self.db.select(database)
        self.db.flushdb()

    def delete_keys(self, keys: list[str], database: int):
        self.db.select(database)
        self.db.delete(*keys)

    def __del__(self):
        print("Saving database to disk...")
        self.db.save()
