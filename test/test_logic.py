from typing import Any, get_type_hints
import unittest
from src.utils.db import Database
from src.config import (
    BLOCK_DB_IDX,
    NODE_COORDS_DB_IDX,
    ABODE_DB_IDX,
    VOTER_DB_IDX,
    Block,
    Abode,
    Voter,
    WriteablePoint,
)


class TestDatabase(unittest.TestCase):
    def setUpClass(self):
        self.db = Database()

    def is_valid_typed_dict(self, instance: Any, typed_dict_cls: type) -> bool:
        """
        Check if the instance is a valid instance of the provided TypedDict class.

        Parameters
        ----------
        instance : Any
            The instance to check
        typed_dict_cls : type
            The TypedDict class to check against

        Returns
        -------
        bool
            Whether the instance is a valid instance of the provided TypedDict class
        """
        type_hints = get_type_hints(typed_dict_cls)
        required_keys = {
            key
            for key, value in type_hints.items()
            if not str(value).startswith("typing_extensions.NotRequired")
        }
        optional_keys = {
            key
            for key, value in type_hints.items()
            if str(value).startswith("typing_extensions.NotRequired")
        }
        all_keys = required_keys | optional_keys

        return required_keys.issubset(instance.keys()) and set(
            instance.keys()
        ).issubset(all_keys)

    def test_db_types(self):
        """Ensure that all values in the database are of the correct type"""
        # Test that all blocks are valid
        for block in self.db.get_all_dict(BLOCK_DB_IDX).values():
            self.assertTrue(
                self.is_valid_typed_dict(block, Block),
                f"Block {block} is not a valid Block",
            )

        # Test that all abodes are valid
        for abode in self.db.get_all_dict(ABODE_DB_IDX).values():
            self.assertTrue(
                self.is_valid_typed_dict(abode, Abode),
                f"Abode {abode} is not a valid Abode",
            )

        # Test that all voters are valid
        for voter in self.db.get_all_dict(VOTER_DB_IDX).values():
            self.assertTrue(
                self.is_valid_typed_dict(voter, Voter),
                f"Voter {voter} is not a valid Person",
            )

        # Test that all node coords are valid
        for node_coords in self.db.get_all_dict(NODE_COORDS_DB_IDX).values():
            self.assertTrue(
                self.is_valid_typed_dict(node_coords, WriteablePoint),
                f"Node coords {node_coords} is not a valid Point",
            )

    def test_node_existence(self):
        """Ensure that the nodes in the block IDs exist in the node coords database"""
        for block_id in self.db.get_keys(BLOCK_DB_IDX):
            node_1, node_2, _ = block_id.split(":")

            self.assertTrue(
                self.db.exists(node_1, NODE_COORDS_DB_IDX),
                f"The coordinates for node {node_1} do not exist in the database",
            )
            self.assertTrue(
                self.db.exists(node_2, NODE_COORDS_DB_IDX),
                f"The coordinates for node {node_2} do not exist in the database",
            )

    def test_block_abode_relation(self):
        """For each block, ensure each abode exists"""
        for block in self.db.get_all_dict(BLOCK_DB_IDX).values():
            for abode_id in block["abodes"]:
                self.assertTrue(
                    self.db.exists(abode_id, ABODE_DB_IDX),
                    f"The abode {abode_id} does not exist in the database",
                )

    def test_abode_block_relation(self):
        """For each abode, ensure the block exists"""
        for abode in self.db.get_all_dict(ABODE_DB_IDX).values():
            self.assertTrue(
                self.db.exists(abode["block_id"], BLOCK_DB_IDX),
                f"The block {abode['block_id']} does not exist in the database",
            )

    def test_abode_voter_relation(self):
        """For each abode, ensure each voter exists"""
        for abode in self.db.get_all_dict(ABODE_DB_IDX).values():
            if "voters" not in abode:
                continue
            elif isinstance(abode["voters"], list):
                for voter_id in abode["voters"]:
                    self.assertTrue(
                        self.db.exists(voter_id, VOTER_DB_IDX),
                        f"The voter {voter_id} does not exist in the database",
                    )
            else:
                # If voters is a dict, the key, value pairs are apt numbers to lists of voters
                for voters in abode["voters"].values():
                    for voter_id in voters:
                        self.assertTrue(
                            self.db.exists(voter_id, VOTER_DB_IDX),
                            f"The voter {voter_id} does not exist in the database",
                        )

    def test_voter_abode_relation(self):
        """For each voter, ensure the abode exists"""
        for voter in self.db.get_all_dict(VOTER_DB_IDX).values():
            self.assertTrue(
                self.db.exists(voter["abode"], ABODE_DB_IDX),
                f"The abode {voter['abode']} does not exist in the database",
            )
