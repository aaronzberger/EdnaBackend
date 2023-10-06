"""
Temporary DB Documentation
--------------------------

Node Coords DB (NODE_COORDS_DB_IDX)
    Key: Node ID (from OSM)
    Value: Dict with keys 'lat' and 'lon'

Blocks (BLOCK_DB_IDX):
    Key: Block ID (from pre-processing, the concatenation of end node IDs and an identifier)
    Value: Block object

"""

import json
from typing import Awaitable
import redis


class Database:
    def __init__(self):
        self.db = redis.Redis(host='redis-container', port=6379, password='votefalcon12', decode_responses=True)

        if not self.db.ping():
            raise ConnectionError('Redis server failed to respond to ping.')

    # @staticmethod
    # def db_select(func):
    #     def wrapper(key, value, database: int):
    #         self.db.select(database: int)
    #         # Wrap in try except?
    #         func(key, value, database: int)

    #     return wrapper

    def set_dict(self, key: str, value: dict, database: int):
        self.set_str(key, json.dumps(value), database)

    def get_dict(self, key: str, database: int) -> dict:
        return json.loads(self.get_str(key, database))

    def set_str(self, key: str, value: str, database: int):
        self.db.select(database)
        self.db.set(key, value)

    def get_str(self, key: str, database: int) -> str:
        self.db.select(database)
        return self.db.get(key)

    def add_to_list(self, key: str, value: str, database: int):
        self.db.select(database)
        self.db.lpush(key, value)

    def get_list(self, key: str, database: int):
        self.db.select(database)
        return self.db.lrange(key, 0, -1)

    def get_list_length(self, key: str, database: int) -> Awaitable[int] | int:
        self.db.select(database)
        return self.db.llen(key)

    def add_to_set(self, key: str, value: str, database: int):
        self.db.select(database)
        self.db.sadd(key, value)

    def get_set(self, key: str, database: int):
        self.db.select(database)
        return self.db.smembers(key)

    def get_set_length(self, key: str, database: int) -> Awaitable[int] | int:
        self.db.select(database)
        return self.db.scard(key)

    def is_in_set(self, key: str, value: str, database: int) -> Awaitable[bool] | bool:
        self.db.select(database)
        return self.db.sismember(key, value)

    def exists(self, key: str, database: int) -> bool:
        self.db.select(database)
        return self.db.exists(key) > 0  # type: ignore

    def delete(self, key: str, database: int):
        self.db.select(database)
        self.db.delete(key)

    def get_keys(self, database: int):
        self.db.select(database)
        return self.db.keys()

    def get_multiple(self, keys: list[str], database: int):
        self.db.select(database)
        values = self.db.mget(keys)
        return dict(zip(keys, [json.loads(value) for value in values]))

    def get_type(self, key: str, database: int):
        self.db.select(database)
        return self.db.type(key)

    def clear_db(self, database: int):
        self.db.select(database)
        self.db.flushdb()

    def __del__(self):
        print('Saving database to disk...')
        self.db.save()
