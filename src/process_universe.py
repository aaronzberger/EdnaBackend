"""Associate PA voter file addresses with the geographic data, determine voter and house values,
and save the results to be used in the optimizer.

A universe file is provided, for which addresses are from the PA voter export.

These addresses must be matched with addresses from the PA address points file
(which were previously associated with blocks in make_blocks.py).
"""
import csv
import dataclasses
import json
import os
import pprint
import sys
from copy import deepcopy
from datetime import datetime
import uuid

import jsonpickle
from rapidfuzz import fuzz, process
from termcolor import colored
from tqdm import tqdm

from src.utils.address import Address, addresses_file_t
from src.config import (
    UUID_NAMESPACE,
    HouseInfo,
    HousePeople,
    Person,
    addresses_file,
    blocks_file,
    blocks_file_t,
    house_to_voters_file,
    manual_match_input_file,
    manual_match_output_file,
    requested_blocks_file,
    street_suffixes_file,
    voters_file_t,
    voter_file_mapping,
    turnout_predictions_file,
    voter_value,
    house_value,
    house_id_to_block_id_file
)


def address_match_score(s1: Address, s2: Address, threshold=90, score_cutoff=0.0):
    """
    Compute a custom score based on the Jaro distance between words in the two strings.

    Parameters
    ----------
        s1 (Address): The first address to compare.
        s2 (Address): The second address to compare.
        threshold (int): The minimum allowed Jaro score for two words to be considered a match.
        score_cutoff (float): The minimum overall score for the function to return a non-zero result.

    Returns
    -------
        A score representing the ratio of matched words to the total number of words.

    Notes
    -----
        Returns 0.0 immediately if the computed score is below score_cutoff or house numbers don't match.
    """

    if s1.house_number != s2.house_number:
        return 0
    if s1.street_name and s2.street_name:
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
    return 0.0


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
        with open("houses_pretty.txt", "w") as f:
            f.write(pprint.pformat(self.addresses_to_id))
        self.need_manual_review = []
        self.manual_matches = json.load(open(manual_match_output_file))
        self.result_counter = {
            "exact match": 0,
            "manual match": 0,
            "does not exist": 0,
            "not in universe": 0,
            "key error": 0,
            "failed": 0,
        }

    def search_manual_associations(
        self, address: Address
    ) -> tuple[Address, str] | None:
        for match in self.manual_matches:
            if match["universe"] == dataclasses.asdict(address):
                if match["match"] == "DNE":
                    self.result_counter["does not exist"] += 1
                    return None
                elif match["match"] == {}:
                    self.result_counter["not in universe"] += 1
                    return None
                else:
                    if isinstance(match["match"], dict):
                        matched_dict: dict = match["match"]
                        as_address = Address(
                            matched_dict["house_number"],
                            matched_dict["house_number_suffix"],
                            matched_dict["street_name"],
                            matched_dict["unit_num"],
                            matched_dict["city"],
                            matched_dict["state"],
                            matched_dict["zip_code"],
                        )
                        if as_address in self.addresses_to_id:
                            self.result_counter["manual match"] += 1
                            return (as_address, match["unit_num"])
                        else:
                            self.result_counter["key error"] += 1
        return None

    def associate(self, address: Address) -> tuple[str, str, str] | None:
        """
        Associate an address with a house and get the house info.

        Parameters
        ----------
            address (str): the address to associate

        Returns
        -------
            HouseInfo | None: information on the house, or None if the address could not be associated
        """

        matched_uuid: str | None = None
        matched_block_id: str | None = None
        custom_unit_num = ""
        choices = []
        precise_match_found = False
        if address in self.addresses_to_id:
            precise_match_found = True
            matched_block_id, matched_uuid = self.addresses_to_id[address]
        else:
            for choice in process.extract_iter(
                query=address,
                choices=self.addresses_to_id.keys(),
                scorer=address_match_score,
                score_cutoff=85,
            ):
                matched_block_id, matched_uuid = self.addresses_to_id[choice[0]]
                if isinstance(choice[0], Address) and (
                    choice[0].unit_num == address.unit_num
                    and choice[0].zip_code == address.zip_code
                    and choice[0].house_number_suffix == address.house_number_suffix
                ):
                    precise_match_found = True
                    self.result_counter["exact match"] += 1

                    # TODO: Handle what happens if there is more than one match here
                else:
                    choices.append(
                        (
                            dataclasses.asdict(choice[0]),
                            choice[1],
                            choice[2],
                            matched_block_id,
                            matched_uuid,
                        )
                    )

        if not precise_match_found:
            result = self.search_manual_associations(address)
            if result:
                matched_block_id, matched_uuid = self.addresses_to_id[result[0]]
                if result[1] != "":
                    custom_unit_num = result[1]
            else:
                self.need_manual_review.append(
                    {"universe": dataclasses.asdict(address), "choices": choices}
                )
                self.result_counter["failed"] += 1
                return None
        if matched_block_id and matched_uuid:
            return matched_block_id, matched_uuid, custom_unit_num
        else:
            return None


def handle_universe_file(
    universe_file: str, blocks: blocks_file_t, turnouts: dict[str, float]
):
    universe_file_opened = open(universe_file)
    num_voters = -1
    for _ in universe_file_opened:
        num_voters += 1
    universe_file_opened.seek(0)
    reader = csv.DictReader(universe_file_opened)

    associater = Associater()

    requested_blocks: blocks_file_t = {}
    requested_voters: voters_file_t = {}

    # To check for duplicate IDs
    voter_ids: set[str] = set()

    # Read in house_id_to_block_id_file to write the new entries
    house_id_to_block_id: dict[str, str] = json.load(
        open(house_id_to_block_id_file)
    )

    def add_voter(
        universe_row: dict, uuid: str, block_id: str, custom_unit_num, custom_uuid
    ):
        """
        Add a voter to the requested voters dictionary.

        Parameters
        ----------
            universe_row (dict): the row from the universe file
            uuid (str): the uuid of the voter

        Returns
        -------
            bool: whether or not the voter was added
        """
        if universe_row["ID Number"] in voter_ids:
            print(
                colored(
                    f"Duplicate ID {universe_row['ID Number']} found. Quitting.", "red"
                )
            )
            sys.exit(1)
        voter_ids.add(universe_row["ID Number"])

        party = (
            "D"
            if universe_row["Party Code"] == "D"
            else ("R" if universe_row["Party Code"] == "R" else "I")
        )

        try:
            turnout: float = turnouts[universe_row["ID Number"]]
        except KeyError:
            print(
                colored(
                    f"Could not find turnout prediction for ID {universe_row['ID']}. Quitting.",
                    "red}",
                )
            )
            sys.exit()

        value = voter_value(party=party, turnout=turnout)

        if value == 0:
            # If we do not want to visit this voter at all, do not add them
            return False

        if custom_uuid not in requested_voters:
            house_info: HouseInfo = blocks[block_id]["addresses"][uuid]
            # TODO: When the manual association returns a unit number, create a new entry in requested_blocks with that unit number
            if custom_unit_num != "":
                requested_voters[custom_uuid] = HousePeople(
                    # TODO: instead of using display_address, use the custom unit number properly
                    # TODO: actually check if there is a custom unit num instead of always showing this lol
                    display_address=f"{house_info['display_address']} Unit {custom_unit_num}",
                    city=house_info["city"],
                    state=house_info["state"],
                    zip=house_info["zip"],
                    latitude=house_info["lat"],
                    longitude=house_info["lon"],
                    voter_info=[],
                    value=-1,
                )
            else:
                requested_voters[custom_uuid] = HousePeople(
                    # TODO: instead of using display_address, use the custom unit number properly
                    # TODO: actually check if there is a custom unit num instead of always showing this lol
                    display_address=house_info["display_address"],
                    city=house_info["city"],
                    state=house_info["state"],
                    zip=house_info["zip"],
                    latitude=house_info["lat"],
                    longitude=house_info["lon"],
                    voter_info=[],
                    value=-1,
                )

        name = f"{universe_row['First Name']} {universe_row['Last Name']}"
        if universe_row["Suffix"] != "":
            name += f" {universe_row['Suffix']}"

        age = (
            datetime.now() - datetime.strptime(universe_row["DOB"], "%m/%d/%Y")
        ).days // 365

        voting_history = {}
        for row_key, election_key in voter_file_mapping.items():
            vote_key = universe_row[f"Election {row_key} Vote Method"]
            voted: bool = vote_key != ""
            by_mail: bool = vote_key != "" and vote_key in ["AB", "MB"]

            voting_history[election_key.name] = voted
            voting_history[election_key.name + "_mail"] = by_mail

        person = Person(
            name=name,
            age=age,
            party=party,
            voting_history=voting_history,
            voter_id=universe_row["ID Number"],
            value=value,
            turnout=turnout,
        )

        requested_voters[custom_uuid]["voter_info"].append(person)

        # Re-calculate the house value with the updated person
        requested_voters[custom_uuid]["value"] = house_value(
            [person["value"] for person in requested_voters[custom_uuid]["voter_info"]]
        )

        return True

    # Process each requested house
    for entry in tqdm(
        reader,
        total=num_voters,
        desc="Processing universe file",
        unit="voters",
        colour="green",
    ):
        if (
            "Address" not in entry
            and "House Number" in entry
            and "Street Name" in entry
        ):
            street_name = Address.sanitize_street_name(entry["Street Name"])

            formatted_address = Address(
                entry["House Number"],
                entry["House Number Suffix"],
                street_name,
                entry["Apartment Number"],
                None,
                None,
                entry["Zip"],
            )

        else:
            raise ValueError(
                "The universe file must contain either an 'Address' column or 'House Number' and 'Street Name' columns"
            )

        result = associater.associate(formatted_address)
        if result is not None:
            block_id, house_uuid, custom_unit_num = result

            # TODO: Currently, this puts everyone whos uuid is the same in the same house,
            # but people can have different apartment numbers. Add a new entry to requested_blocks
            # with a new uuid with the unit number, and use this. Post-processing only uses these two files anyway.

            if custom_unit_num == "":
                custom_uuid = house_uuid
            else:
                custom_uuid = str(
                    uuid.uuid5(
                        UUID_NAMESPACE, str(blocks[block_id]["addresses"][house_uuid])
                    )
                )

                house_id_to_block_id[custom_uuid] = block_id

            added: bool = add_voter(
                entry, house_uuid, block_id, custom_unit_num, custom_uuid
            )

            if not added:
                continue

            if block_id in requested_blocks:
                requested_blocks[block_id]["addresses"][custom_uuid] = blocks[block_id][
                    "addresses"
                ][house_uuid]

            else:
                requested_blocks[block_id] = deepcopy(blocks[block_id])
                requested_blocks[block_id]["addresses"] = {
                    custom_uuid: blocks[block_id]["addresses"][house_uuid]
                }

            requested_blocks[block_id]["addresses"][custom_uuid][
                "value"
            ] = requested_voters[custom_uuid]["value"]

    print(
        colored(
            f"Of {num_voters} voters, {num_voters - associater.result_counter['failed']} were matched",
            "green",
        )
    )
    print(
        colored(
            f"Of the remaining {associater.result_counter['failed']}, {associater.result_counter['does not exist']} do not exist,"
            + f"and {associater.result_counter['not in universe'] + associater.result_counter['key error']} are not in the universe",
            "yellow",
        )
    )

    # no_choices = len(list(x for x in associater.need_manual_review if len(x["choices"]) == 0))
    # print(f"Number of failed houses with no matches at all: {no_choices}")
    with open(manual_match_input_file, "w") as manual_match_file:
        json.dump(associater.need_manual_review, manual_match_file)

    # Write the requested blocks and voters to files
    with open(requested_blocks_file, "w") as f:
        json.dump(requested_blocks, f)
    with open(house_to_voters_file, "w") as f:
        json.dump(requested_voters, f)

    # Write the house_id_to_block_id file
    with open(house_id_to_block_id_file, "w") as f:
        json.dump(house_id_to_block_id, f)


if __name__ == "__main__":
    all_blocks: blocks_file_t = json.load(open(blocks_file))

    turnouts: dict[str, float] = json.load(open(turnout_predictions_file))

    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("Usage: python3 associate.py <universe_file>")
        sys.exit(1)

    handle_universe_file(sys.argv[1], all_blocks, turnouts)
