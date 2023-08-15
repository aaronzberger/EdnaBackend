"""Associate PA voter file addresses with PA address points file addresses:

A universe file is provided, for which addresses are from the PA voter export.

These addresses must be matched with addresses from the PA address points file
(which were previously associated with blocks in make_blocks.py).
"""

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

        if s1.house_number == s2.house_number:
            if s1.zip_code == s2.zip_code:
                return 100
            else:
                return 0
        else:
            return 0

        s1_words = s1.split()
        s2_words = s2.split()

        # Extract house numbers using regular expressions
        s1_house_number = [int(s) for s in s1_words if s.isdigit()]
        s2_house_number = [int(s) for s in s2_words if s.isdigit()]

        # If either address doesn't have a house number or if they don't match, return 0.0
        if not (s1_house_number and s2_house_number):
            return 0.0
        if s1_house_number[0] != s2_house_number[0]:
            return 0.0

        # Remove house numbers from the addresses
        s1 = s1.replace(str(s1_house_number), "", 1).strip()
        s2 = s2.replace(str(s2_house_number), "", 1).strip()

        whole_str_ratio = fuzz.ratio(s1, s2)
        if whole_str_ratio > threshold and whole_str_ratio > score_cutoff:
            return whole_str_ratio

        if len(s1_words) > 1:
            s1_words.pop()

        if len(s2_words) > 1:
            s2_words.pop()

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

        matched_uuid = ""

        if address in self.addresses_to_id:
            matched_block_id, matched_uuid = self.addresses_to_id[address]
        else:
            print(f"Failed to find exact match for address: {address}")
            for choice in process.extract_iter(
                query=address,
                choices=self.addresses_to_id.keys(),
                scorer=self.address_match_score,
                score_cutoff=45,
            ):
                print(f"Choice for fuzzy match: {choice}")
            self.failed_houses += 1
            return None

        return (matched_block_id, matched_uuid)
