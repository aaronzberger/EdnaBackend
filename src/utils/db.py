from typing import Awaitable
import redis


class Database:
    @classmethod
    def __init__(cls):
        cls.db = redis.Redis(host='redis-container', port=6379, password='votefalcon12', decode_responses=True)

        if not Database.db.ping():
            raise ConnectionError('Redis server failed to respond to ping.')

    # @staticmethod
    # def db_select(func):
    #     def wrapper(key, value, database: int):
    #         cls.db.select(database: int)
    #         # Wrap in try except?
    #         func(key, value, database: int)

    #     return wrapper

    @classmethod
    def set_dict(cls, key: str, value: dict, database: int):
        cls.db.select(database)
        cls.db.hset(key, mapping=value)

    @classmethod
    def get_dict(cls, key: str, database: int) -> dict:
        cls.db.select(database)
        return cls.db.hgetall(key)

    @classmethod
    def set_str(cls, key: str, value: str, database: int):
        cls.db.select(database)
        cls.db.set(key, value)

    @classmethod
    def get_str(cls, key: str, database: int) -> str:
        cls.db.select(database)
        return cls.db.get(key)

    @classmethod
    def add_to_list(cls, key: str, value: str, database: int):
        cls.db.select(database)
        cls.db.lpush(key, value)

    @classmethod
    def get_list(cls, key: str, database: int):
        cls.db.select(database)
        return cls.db.lrange(key, 0, -1)

    @classmethod
    def get_list_length(cls, key: str, database: int) -> Awaitable[int] | int:
        cls.db.select(database)
        return cls.db.llen(key)

    @classmethod
    def exists(cls, key: str, database: int) -> bool:
        cls.db.select(database)
        return cls.db.exists(key) > 0  # type: ignore

    @classmethod
    def delete(cls, key: str, database: int):
        cls.db.select(database)
        cls.db.delete(key)

    @classmethod
    def get_keys(cls, database: int):
        cls.db.select(database)
        return cls.db.keys()

    @classmethod
    def __del__(cls):
        print('Saving database to disk...')
        cls.db.save()
