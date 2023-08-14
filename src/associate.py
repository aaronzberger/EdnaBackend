"""Associate PA voter file addresses with PA address points file addresses:

A universe file is provided, for which addresses are from the PA voter export.

These addresses must be matched with addresses from the PA address points file
(which were previously associated with blocks in make_blocks.py).
"""

from typing import Optional
from copy import deepcopy
from src.config import (
    HouseInfo,
    houses_file_t,
    houses_file,
    blocks_file,
    blocks_file_t,
    street_suffixes_file,
)

import re

from rapidfuzz import process, fuzz

import json


class Associater:
    all_blocks: blocks_file_t = json.load(open(blocks_file))
    houses_to_id: houses_file_t = json.load(open(houses_file))
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

        self.sanitized_houses_to_id = {
            self.sanitize_address(key): value
            for key, value in self.houses_to_id.items()
        }

    def address_match_score(self, s1: str, s2: str, threshold=90, score_cutoff=0.0):
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

    def associate(self, address: str) -> tuple[str, HouseInfo] | None:
        """
        Associate an address with a house and get the house info

        Parameters:
            address (str): the address to associate

        Returns:
            HouseInfo | None: information on the house, or None if the address could not be associated
        """
        if address in self.houses_to_id:
            block_id = self.houses_to_id[address]
        else:
            print(f"Failed to find exact match for address: {address}")
            for choice in process.extract_iter(
                query=self.sanitize_address(address),
                choices=self.sanitized_houses_to_id.keys(),
                scorer=self.address_match_score,
                score_cutoff=45,
            ):
                print(f"Choice for fuzzy match: {choice}")
            self.failed_houses += 1
            return None

        return (block_id, deepcopy(self.all_blocks[block_id]["addresses"][address]))
