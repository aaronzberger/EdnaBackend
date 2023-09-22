"""Process the universe and output a list based on some criteria."""

from copy import deepcopy
import csv
import json
import os
import sys
from typing import NamedTuple

from termcolor import colored
from tqdm import tqdm

from src.utils.address import Address
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
from src.utils.viz import display_targeting_voters


class AddressAndHouse(NamedTuple):
    address: Address
    people: list[Person]
    address_line_1: str
    address_line_2: str
    coords: Point


stats = {
    "num_voters": 0,
    "num_sending": 0,
    "num_indeps": 0,
    "num_dems": 0,
    "num_reps": 0,
    "num_too_unlikely": 0,
    "num_houses": 0,
}

# VISTA_FIELDNAMES = ["Salutation", "First name", "Middle", "Last name (Required if no Company)", "Suffix", "Title",
#                     "Company (Required if no Last name)", "Address Line 1 (Required)", "Address Line 2",
#                     "City (Required)", "State (Required)", "Zip Code (Required. 5- or 9- digits)"]

VISTA_FIELDNAMES = ["Salutation", "First name", "Middle", "Name", "Suffix", "Title",
                    "Company (Required if no Last name)", "Address Line 1 (Required)", "Address Line 2",
                    "City (Required)", "State (Required)", "Zip Code (Required. 5- or 9- digits)"]


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

    output_file_d = open(os.path.join(BASE_DIR, "full_pool_d.csv"), "w")
    writer_d = csv.DictWriter(
        output_file_d, VISTA_FIELDNAMES
    )
    writer_d.writeheader()

    output_file_r = open(os.path.join(BASE_DIR, "full_pool_r.csv"), "w")
    writer_r = csv.DictWriter(
        output_file_r, VISTA_FIELDNAMES
    )
    writer_r.writeheader()

    output_file_i = open(os.path.join(BASE_DIR, "full_pool_i.csv"), "w")
    writer_i = csv.DictWriter(
        output_file_i, VISTA_FIELDNAMES
    )
    writer_i.writeheader()

    output_file_store = open(os.path.join(BASE_DIR, "full_pool_store.csv"), "w")
    store_writer = csv.DictWriter(
        output_file_store, fieldnames=VISTA_FIELDNAMES + ["PA-ID"]
    )
    store_writer.writeheader()

    voters: list[AddressAndHouse] = []
    previously_inserted: set[Address] = set()

    def add_voter(universe_row: dict, address: Address):
        global stats
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
            turnout: float = turnouts[universe_row["ID Number"]]
        except KeyError:
            print(
                colored(
                    f"Could not find turnout prediction for ID {universe_row['ID Number']}. Quitting.",
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

        stats["num_voters"] += 1

        # Criteria for adding a voter:
        if (party == "I" and turnout < 0.2) or (party in ["D", "R"] and turnout < 0.35):
            stats["num_too_unlikely"] += 1
        else:
            stats["num_sending"] += 1
            if party == "D":
                stats["num_dems"] += 1
            elif party == "R":
                stats["num_reps"] += 1
            else:
                stats["num_indeps"] += 1

            if address in previously_inserted:
                # Insert this name into the previous address
                for voter in voters:
                    if voter.address == address:
                        voter.people.append(
                            Person(
                                name=name.casefold().title(),
                                age=0,
                                party=party,
                                voting_history={},
                                value=turnout,
                                turnout=turnout,
                                voter_id=universe_row["ID Number"],
                            )
                        )
                        return False
            else:
                stats["num_houses"] += 1
                previously_inserted.add(address)

            voters.append(
                AddressAndHouse(
                    address=address,
                    people=[
                        Person(
                            name=name.casefold().title(),
                            age=0,
                            party=party,
                            voting_history={},
                            value=turnout,
                            turnout=turnout,
                            voter_id=universe_row["ID Number"],
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
                entry["City"],
                entry["State"],
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
            block_id, uuid, _ = result
            voter.coords["lat"] = blocks[block_id]["addresses"][uuid]["lat"]
            voter.coords["lon"] = blocks[block_id]["addresses"][uuid]["lon"]

        combined_names = ", ".join([person["name"] for person in voter.people])
        combined_ids = ", ".join([person["voter_id"] for person in voter.people])

        # Use the output fieldnames to write the row
        output = {
            "Salutation": "",
            "First name": "",
            "Middle": "",
            "Name": combined_names,
            "Suffix": "",
            "Title": "",
            "Company (Required if no Last name)": "",
            "Address Line 1 (Required)": voter.address_line_1,
            "Address Line 2": "",
            "City (Required)": voter.address.city,
            "State (Required)": voter.address.state,
            "Zip Code (Required. 5- or 9- digits)": voter.address.zip_code,
        }
        rows_written = 0

        # If any voter is independent, send to the independent list
        for person in voter.people:
            if person["party"] == "I":
                writer_i.writerow(output)
                rows_written += 1
                break

        if rows_written == 1:
            continue

        for person in voter.people:
            if person["party"] == "D":
                writer_d.writerow(output)
                rows_written += 1
                break

        if rows_written == 1:
            continue

        for person in voter.people:
            if person["party"] == "R":
                writer_r.writerow(output)
                rows_written += 1
                break

        assert rows_written == 1

        store_output = deepcopy(output)
        store_output["PA-ID"] = combined_ids
        store_writer.writerow(store_output)

    display_targeting_voters(voters).save(os.path.join(BASE_DIR, "viz", "full_pool.html"))

    print(f"Of {stats['num_voters']} voters:")
    print(f"  {stats['num_sending']} ({stats['num_sending'] / stats['num_voters'] * 100:.2f}%) are being sent to")
    print(f"  {stats['num_houses']} are unique houses")
    print(f"  {stats['num_indeps']} ({stats['num_indeps'] / stats['num_sending'] * 100:.2f}%) are independents")
    print(f"  {stats['num_dems']} ({stats['num_dems'] / stats['num_sending'] * 100:.2f}%) are democrats")
    print(f"  {stats['num_reps']} ({stats['num_reps'] / stats['num_sending'] * 100:.2f}%) are republicans")
    print(
        f"  {stats['num_too_unlikely']} ({stats['num_too_unlikely'] / stats['num_voters'] * 100:.2f}%) are too unlikely to vote"
    )

    output_file_d.close()
    output_file_r.close()
    output_file_i.close()


if __name__ == "__main__":
    all_blocks: blocks_file_t = json.load(open(blocks_file))

    turnouts: dict[str, float] = json.load(open(turnout_predictions_file))

    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("Usage: python3 mail_list.py <universe_file>")
        sys.exit(1)

    handle_universe_file(sys.argv[1], all_blocks, turnouts)
