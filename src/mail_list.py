"""Process the universe and output a list based on some criteria."""

import csv
import json
import os
import sys
from typing import NamedTuple, Optional

from termcolor import colored
from tqdm import tqdm

from src.address import Address
from src.config import (
    BASE_DIR,
    NodeType,
    Person,
    Point,
    blocks_file,
    blocks_file_t,
    turnout_predictions_file,
)
from src.process_universe import Associater
from src.viz_utils import display_targeting_voters


class AddressAndHouse(NamedTuple):
    address: Address
    people: list[Person]
    address_line_1: str
    address_line_2: str
    coords: Point


num_invalid_indeps = 0
num_indeps = 0
num_voters = 0
num_houses = 0
num_sending_voters = 0


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

    output_file = open(os.path.join(BASE_DIR, "indep_addresses.csv"), "w")
    writer = csv.DictWriter(
        output_file, fieldnames=["Name", "Address Line 1", "Address Line 2", "State", "Zip Code", "PA-ID"]
    )
    writer.writeheader()

    voters: list[AddressAndHouse] = []
    previously_inserted: set[Address] = set()

    def add_voter(universe_row: dict, address: Address):
        global num_voters, num_indeps, num_invalid_indeps, num_houses, num_sending_voters
        """
        Add a voter to the requested voters dictionary.

        Parameters
        ----------
            universe_row (dict): the row from the universe file
            uuid (str): the uuid of the voter

        Returns
        -------
            bool: whether or not a new address was added
        """
        party = (
            "D"
            if universe_row["Party Code"] == "D"
            else ("R" if universe_row["Party Code"] == "R" else "I")
        )

        try:
            turnout: float = turnouts[universe_row["ID"]]
        except KeyError:
            print(
                colored(
                    f"Could not find turnout prediction for ID {universe_row['ID']}. Quitting.",
                    "red}",
                )
            )
            sys.exit()

        name = f"{universe_row['First Name']} {universe_row['Last Name']}"
        if universe_row["Suffix"] != "":
            name += f" {universe_row['Suffix']}"

        address_line_1 = f"{universe_row['House Number']}"
        if universe_row["House Number Suffix"] != "":
            address_line_1 += f" {universe_row['House Number Suffix']}"
        address_line_1 += f" {universe_row['Street Name']}"
        if universe_row["Apartment Number"] != "":
            address_line_1 += f" {universe_row['Apartment Number']}"
        address_line_2 = f"{universe_row['City']}, PA {universe_row['Zip']}"

        num_voters += 1

        if party == "I":
            num_indeps += 1
        if party == "I" and turnout < 0.2:
            num_invalid_indeps += 1

        # Criteria for adding a voter:
        if party == "I" and turnout >= 0.2:
            num_sending_voters += 1
            if address in previously_inserted:
                # Insert this name into the previous address
                for voter in voters:
                    if voter.address == address:
                        voter.people.append(
                            Person(
                                name=name.casefold().title(),
                                age=0,
                                party="I",
                                voting_history={},
                                value=turnout,
                                voter_id=universe_row["ID"],
                            )
                        )
                        return False
            else:
                num_houses += 1
                previously_inserted.add(address)

            voters.append(
                AddressAndHouse(
                    address=address,
                    people=[
                        Person(
                            name=name.casefold().title(),
                            age=0,
                            party="I",
                            voting_history={},
                            value=turnout,
                            voter_id=universe_row["ID"],
                        )
                    ],
                    address_line_1=address_line_1,
                    address_line_2=address_line_2,
                    coords=Point(lat=-1, lon=-1, type=NodeType.other, id=""),
                )
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

        add_voter(entry, formatted_address)

    for voter in tqdm(voters, desc="Associating", unit="voters", colour="green"):
        result = associater.associate(voter.address)
        if result is not None:
            block_id, uuid = result
            voter.coords["lat"] = blocks[block_id]["addresses"][uuid]["lat"]
            voter.coords["lon"] = blocks[block_id]["addresses"][uuid]["lon"]

        combined_names = ", ".join([person["name"] for person in voter.people])
        combined_ids = ", ".join([person["voter_id"] for person in voter.people])

        writer.writerow(
            {
                "Name": combined_names,
                "Address Line 1": voter.address_line_1,
                "Address Line 2": voter.address_line_2,
                "PA-ID": combined_ids,
            }
        )

    display_targeting_voters(voters).save(os.path.join(BASE_DIR, "viz", "indeps.html"))

    print(f"Of {num_voters} voters, {num_indeps} are independents, and {num_invalid_indeps} of those have too low turnout scores.")
    print(f"There are {num_houses} houses which have {num_sending_voters} voters")

    output_file.close()


if __name__ == "__main__":
    all_blocks: blocks_file_t = json.load(open(blocks_file))

    turnouts: dict[str, float] = json.load(open(turnout_predictions_file))

    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("Usage: python3 mail_list.py <universe_file>")
        sys.exit(1)

    handle_universe_file(sys.argv[1], all_blocks, turnouts)
