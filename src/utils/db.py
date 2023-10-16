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

"""

import json
from typing import Awaitable, Optional
import redis


class Database:
    def __init__(self):
        self.db = redis.Redis(host='redis-container', port=6379, password='votefalcon12', decode_responses=True)

        if not self.db.ping():
            raise ConnectionError('Redis server failed to respond to ping.')

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
        return self.db.get(key)

    def get_multiple_str(self, keys: list[str], database: int) -> list[str]:
        self.db.select(database)
        return self.db.mget(keys)

    def add_to_list(self, key: str, value: str, database: int):
        self.db.select(database)
        self.db.lpush(key, value)

    def get_list(self, key: str, database: int):
        self.db.select(database)
        return self.db.lrange(key, 0, -1)

    def get_list_length(self, key: str, database: int) -> int:
        self.db.select(database)
        return self.db.llen(key)

    def add_to_set(self, key: str, value: str, database: int):
        self.db.select(database)
        self.db.sadd(key, value)

    def get_set(self, key: str, database: int) -> set[str]:
        self.db.select(database)
        return self.db.smembers(key)

    def get_set_length(self, key: str, database: int) -> int:
        self.db.select(database)
        return self.db.scard(key)

    def is_in_set(self, key: str, value: str, database: int) -> bool:
        self.db.select(database)
        return self.db.sismember(key, value)

    def exists(self, key: str, database: int) -> bool:
        self.db.select(database)
        return self.db.exists(key) > 0  # type: ignore

    def delete(self, key: str, database: int):
        self.db.select(database)
        self.db.delete(key)

    def get_keys(self, database: int) -> list[str]:
        self.db.select(database)
        return self.db.keys()

    def get_all(self, database: int):
        """Get a dictionary of (key, value) pairs for all keys in the database."""
        self.db.select(database)
        keys = self.db.keys()
        values = self.db.mget(keys)
        return dict(zip(keys, [None if v is None else json.loads(v) for v in values]))

    def get_multiple(self, keys: list[str], database: int) -> dict[str, dict]:
        self.db.select(database)
        values = self.db.mget(keys)
        return dict(zip(keys, [None if v is None else json.loads(v) for v in values]))
    
    def set_multiple_dict(self, key_value_pairs: dict[str, dict], database: int):
        self.db.select(database)
        casted: dict[str, str] = {str(k): json.dumps(v) for k, v in key_value_pairs.items()}
        self.db.mset(casted)

    def get_type(self, key: str, database: int):
        self.db.select(database)
        return self.db.type(key)

    def clear_db(self, database: int):
        self.db.select(database)
        self.db.flushdb()

    def __del__(self):
        print('Saving database to disk...')
        self.db.save()
