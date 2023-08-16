"""Associate PA voter file addresses with PA address points file addresses:

A universe file is provided, for which addresses are from the PA voter export.

These addresses must be matched with addresses from the PA address points file
(which were previously associated with blocks in make_blocks.py).
"""
import dataclasses
from typing import Optional
from copy import deepcopy

import jsonpickle

from src.config import (
    HouseInfo,
    addresses_file,
    blocks_file,
    blocks_file_t,
    street_suffixes_file,
)

from src.address import Address, addresses_file_t

from rapidfuzz import process, fuzz

import json
import pprint


class Associater:
    all_blocks: blocks_file_t = json.load(open(blocks_file))

    with open(addresses_file) as addresses_file_instance:
        addresses_to_id: addresses_file_t = jsonpickle.decode(
            addresses_file_instance.read(), keys=True
        )
    street_suffixes: dict[str, str] = json.load(open(street_suffixes_file))

    def sanitize_address(self, address: str):
        # Split the street name by spaces
        words = address.casefold().split()

        if len(words) > 1:
            last_word = words[-1]

            # Check if the last word is in the lookup dictionary
            if last_word in self.street_suffixes:
                # If it is, replace it
                words[-1] = self.street_suffixes[last_word]

        # Join the words back together and return
        return " ".join(words).rstrip()

    def __init__(self):
        self.failed_houses = self.apartment_houses = 0
        with open('houses_pretty.txt', 'w') as f:
            f.write(pprint.pformat(self.addresses_to_id))
        self.need_manual_review = []

    def address_match_score(
            self, s1: Address, s2: Address, threshold=90, score_cutoff=0.0
    ):
        """
        Computes a custom score based on the Jaro distance between words in the two strings.

        Args:
        - s1, s2: The two strings to compare.
        - threshold: The minimum allowed Jaro score for two words to be considered a match.
        - score_cutoff: The minimum overall score for the function to return a non-zero result.

        Returns:
        - A score representing the ratio of matched words to the total number of words.
        Returns 0.0 immediately if the computed score is below score_cutoff or house numbers don't match.
        """

        if s1.house_number != s2.house_number:
            return 0

        s1_words = s1.street_name.split()
        s2_words = s2.street_name.split()

        whole_str_ratio = fuzz.ratio(s1.street_name, s2.street_name)
        if whole_str_ratio > threshold and whole_str_ratio > score_cutoff:
            return whole_str_ratio

        matched_words = 0

        for word1 in s1_words:
            for word2 in s2_words:
                # Compute the Jaro distance between word1 and word2
                jaro_score = fuzz.ratio(word1, word2)
                if jaro_score >= threshold:
                    matched_words += 1

        total_words = max(len(s1_words), len(s2_words))

        score = (matched_words / total_words) * 100

        return score if score >= score_cutoff else 0.0

    def associate(self, address: Address) -> tuple[str, str] | None:
        """
        Associate an address with a house and get the house info

        Parameters:
            address (str): the address to associate

        Returns:
            HouseInfo | None: information on the house, or None if the address could not be associated
        """

        matched_uuid: str | None = None
        matched_block_id: str | None = None
        choices = []
        precise_match_found = False
        if address in self.addresses_to_id:
            precise_match_found = True
            matched_block_id, matched_uuid = self.addresses_to_id[address]
        else:
            # print(f"Failed to find exact match for address: {address}")

            for choice in process.extract_iter(
                    query=address,
                    choices=self.addresses_to_id.keys(),
                    scorer=self.address_match_score,
                    score_cutoff=85,
            ):
                matched_block_id, matched_uuid = self.addresses_to_id[choice[0]]
                if isinstance(choice[0], Address) and (
                        choice[0].unit_num == address.unit_num and
                        choice[0].zip_code == address.zip_code and
                        choice[0].house_number_suffix == address.house_number_suffix):
                    precise_match_found = True
                    # print(f"Fuzzy matched\n{address} to \n{choice[0]}")
                    # TODO: Handle what happens if there is more than one match here
                else:
                    choices.append((dataclasses.asdict(choice[0]), choice[1], choice[2], matched_block_id, matched_uuid))
                    # print(f"Choice for fuzzy match: {choice}")

        if not precise_match_found:
            self.need_manual_review.append({"universe": dataclasses.asdict(address), "choices": choices})
            self.failed_houses += 1
            return None
        else:
            return matched_block_id, matched_uuid
