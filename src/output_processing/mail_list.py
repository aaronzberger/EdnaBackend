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
    # blocks_file,
    # blocks_file_t,
    turnout_predictions_file,
    mail_data_file,
    BLOCK_DB_IDX,
    PLACE_DB_IDX,
    VOTER_DB_IDX
)
from src.process_universe import Associater
from src.utils.viz import display_targeting_voters
from src.utils.db import Database


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
    "num_ballot_already_sent": 0,
    "num_too_unlikely": 0,
    "num_houses": 0,
    "num_failed": 0,
}

db = Database()

# # FRANKLIN PK
# PRECINCT_CODES = [
#     "1450101",
#     "1450102",
#     "1450103",
#     "1450201",
#     "1450202",
#     "1450203",
#     "1450301",
#     "1450302",
#     "1450303",
#     "MN145"
# ]

# num_voters = 0
# voter_to_uuid: dict[str, str] = {}
# uuids_to_voters: dict[str, str] = json.load(open(house_to_voters_file))
# for uuid, info in uuids_to_voters.items():
#     for voter in info["voter_info"]:
#         voter_to_uuid[voter["voter_id"]] = uuid
#         num_voters += 1

# print("NUM VOTERS IN UNIVERSE:", num_voters)


# VISTA_FIELDNAMES = ["Salutation", "First name", "Middle", "Last name (Required if no Company)", "Suffix", "Title",
#                     "Company (Required if no Last name)", "Address Line 1 (Required)", "Address Line 2",
#                     "City (Required)", "State (Required)", "Zip Code (Required. 5- or 9- digits)"]

VISTA_FIELDNAMES = ["Salutation", "First name", "Middle", "Name", "Suffix", "Title",
                    "Company (Required if no Last name)", "Address Line 1 (Required)", "Address Line 2",
                    "City (Required)", "State (Required)", "Zip Code (Required. 5- or 9- digits)"]


def handle_universe_file(
    universe_file: str, turnouts: dict[str, float], mail_data: dict[str, dict]
):
    universe_file_opened = open(universe_file)
    num_voters = -1
    for _ in universe_file_opened:
        num_voters += 1
    universe_file_opened.seek(0)
    reader = csv.DictReader(universe_file_opened)

    associater = Associater()

    output_file_d = open(os.path.join(BASE_DIR, "election_day_di.csv"), "w")
    writer_d = csv.DictWriter(
        output_file_d, VISTA_FIELDNAMES
    )
    writer_d.writeheader()

    output_file_r = open(os.path.join(BASE_DIR, "election_day_r.csv"), "w")
    writer_r = csv.DictWriter(
        output_file_r, VISTA_FIELDNAMES
    )
    writer_r.writeheader()

    output_file_store = open(os.path.join(BASE_DIR, "election_day_store.csv"), "w")
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

        # if party == "R":
        #     return False

        # if universe_row["Precinct Code"] not in PRECINCT_CODES:
        #     return False

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

        try:
            mail_ballot_data = mail_data[universe_row["ID Number"]]
        except KeyError:
            mail_ballot_data = {"BallotSent": None}

        if mail_ballot_data["BallotSent"] is not None:
            stats["num_ballot_already_sent"] += 1
            return False

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
        if turnout < 0.65:
            stats["num_too_unlikely"] += 1
            return False

        stats["num_sending"] += 1
        if party == "D":
            stats["num_dems"] += 1
        elif party == "I":
            stats["num_indeps"] += 1
        else:
            stats["num_reps"] += 1

        if address in previously_inserted:
            # Insert this name into the previous address
            for voter in voters:
                if voter.address == address:
                    voter.people.append(
                        Person(
                            name=name.casefold().title(),
                            age=0,
                            party=party,
                            voter_id=universe_row["ID Number"],
                            place="",
                            voting_history={},
                            value=turnout,
                            turnout=turnout,
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
                        place="",
                        voter_id=universe_row["ID Number"],
                        voting_history={},
                        value=turnout,
                        turnout=turnout,
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
            house = db.get_dict(block_id, BLOCK_DB_IDX)["places"][uuid]
            voter.coords["lat"] = house["lat"]
            voter.coords["lon"] = house["lon"]

        names = [person["name"] for person in voter.people]

        # If there are more than 2 voters, find a common last name and use that
        if len(names) > 2:
            last_names = [name.split(' ')[-1] for name in names]
            last_name = max(set(last_names), key=last_names.count)
            display_name = f'{last_name} Family'
        else:
            display_name = ' & '.join(names)

        # combined_names = ", ".join([person["name"] for person in voter.people])
        combined_ids = ", ".join([person["voter_id"] for person in voter.people])

        # Use the output fieldnames to write the row
        output = {
            "Salutation": "",
            "First name": "",
            "Middle": "",
            "Name": display_name,
            "Suffix": "",
            "Title": "",
            "Company (Required if no Last name)": "",
            "Address Line 1 (Required)": voter.address_line_1,
            "Address Line 2": "",
            "City (Required)": voter.address.city,
            "State (Required)": voter.address.state,
            "Zip Code (Required. 5- or 9- digits)": voter.address.zip_code,
        }

        # If any voter is D or I, write to the D file
        if any(person["party"] != "R" for person in voter.people):
            writer_d.writerow(output)
        else:
            writer_r.writerow(output)

        store_output = deepcopy(output)
        store_output["PA-ID"] = combined_ids
        store_writer.writerow(store_output)

    display_targeting_voters(voters).save(os.path.join(BASE_DIR, "viz", "election_day.html"))

    print(f"Of {stats['num_voters']} voters:")
    print(f"  {stats['num_sending']} ({stats['num_sending'] / stats['num_voters'] * 100:.2f}%) are being sent to")
    print(f"  {stats['num_houses']} are unique houses")
    print(f"  {stats['num_indeps']} ({stats['num_indeps'] / stats['num_voters'] * 100:.2f}%) are independents")
    print(f"  {stats['num_dems']} ({stats['num_dems'] / stats['num_voters'] * 100:.2f}%) are democrats")
    print(f"  {stats['num_reps']} ({stats['num_reps'] / stats['num_voters'] * 100:.2f}%) are republicans")
    print(f"  {stats['num_ballot_already_sent']} ({stats['num_ballot_already_sent'] / stats['num_voters'] * 100:.2f}%) have already been sent a ballot")
    print(f"  {stats['num_too_unlikely']} ({stats['num_too_unlikely'] / stats['num_voters'] * 100:.2f}%) are too unlikely to vote")
    print(f"  {stats['num_failed']} ({stats['num_failed'] / stats['num_voters'] * 100:.2f}%) failed to be processed")

    output_file_d.close()
    output_file_r.close()
    output_file_store.close()


if __name__ == "__main__":
    turnouts: dict[str, float] = json.load(open(turnout_predictions_file))

    mail_data: dict[str, dict] = json.load(open(mail_data_file))

    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("Usage: python3 mail_list.py <universe_file>")
        sys.exit(1)

    handle_universe_file(sys.argv[1], turnouts, mail_data)
