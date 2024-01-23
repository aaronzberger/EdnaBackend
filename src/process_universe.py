"""Associate PA voter file addresses with the geographic data, determine voter and house values,
and save the results to be used in the optimizer.

A universe file is provided, for which addresses are from the PA voter export.

These addresses must be matched with addresses from the PA address points file
(which were previously associated with blocks in make_blocks.py).
"""
import csv
import json
import os
import sys
from datetime import datetime
from typing import Optional

from termcolor import colored
from tqdm import tqdm

from src.associate import Associater
from src.config import (
    CAMPAIGN_ID,
    CAMPAIGN_SUBSET_DB_IDX,
    ABODE_DB_IDX,
    VOTER_DB_IDX,
    Voter,
    Abode,
    manual_match_input_file,
    turnout_predictions_file,
    voter_file_mapping,
    voter_value,
)
from src.utils.address import Address
from src.utils.db import Database

db = Database()


def handle_universe_file(universe_file: str, turnouts: dict[str, float]):
    universe_file_opened = open(universe_file)
    num_voters = -1
    for _ in universe_file_opened:
        num_voters += 1
    universe_file_opened.seek(0)
    reader = csv.DictReader(universe_file_opened)

    associater = Associater()

    # Clear this campaign's subset of the database
    db.delete(CAMPAIGN_ID, CAMPAIGN_SUBSET_DB_IDX)

    def add_voter(universe_row: dict, abode_id: str, custom_unit_num: Optional[str]):
        """
        Add a voter to the requested voters dictionary.

        Parameters
        ----------
            universe_row (dict): the row from the universe file
            abode_id (str): theid of the abode this voter is located
            custom_unit_num (Optional[str]): the custom unit number for this voter

        Returns
        -------
            bool: whether or not the voter was added
        """
        voter_id = universe_row["ID Number"]
        if db.is_in_set(CAMPAIGN_ID, voter_id, CAMPAIGN_SUBSET_DB_IDX):
            # This also means the voter has been accounted for in the abode and block databases
            return False

        party = (
            "D"
            if universe_row["Party Code"] == "D"
            else ("R" if universe_row["Party Code"] == "R" else "I")
        )

        try:
            turnout: float = turnouts[voter_id]
        except KeyError:
            print(
                colored(
                    f"Could not find turnout prediction for ID {universe_row['ID']}. Quitting.",
                    "red}",
                )
            )
            sys.exit()

        # Here, we do the only universe-wide elimination
        value = voter_value(party=party, turnout=turnout)

        if value == 0:
            # If we do not want to visit this voter at all, do not add them
            return False

        abode: Abode = db.get_dict(abode_id, ABODE_DB_IDX)

        if custom_unit_num is not None:
            if "voter_ids" not in abode:
                # This is the first voter in this unit
                abode["voter_ids"] = {custom_unit_num: [voter_id]}
            elif type(abode["voter_ids"]) is list:
                # Other voters do not have units, so place them in the default unit
                abode["voter_ids"] = {
                    "": abode["voter_ids"],
                    custom_unit_num: [voter_id],
                }
            elif type(abode["voter_ids"]) is dict:
                # There are already voters in this and/or other units
                if custom_unit_num in abode["voter_ids"]:
                    abode["voter_ids"][custom_unit_num].append(voter_id)
                else:
                    abode["voter_ids"][custom_unit_num] = [voter_id]
        else:
            if "voter_ids" not in abode:
                # This is the first voter in this unit
                abode["voter_ids"] = [voter_id]
            elif type(abode["voter_ids"]) is list:
                abode["voter_ids"].append(voter_id)
            elif type(abode["voter_ids"]) is dict:
                # There are already voters in this and/or other units, so place this voter in the default unit
                if "" in abode["voter_ids"]:
                    abode["voter_ids"][""].append(voter_id)
                else:
                    abode["voter_ids"][""] = [voter_id]

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

        voter = Voter(
            id=voter_id,
            name=name,
            age=age,
            party=party,
            abode_id=abode_id,
            voting_history=voting_history,
            turnout=turnout,
        )

        if custom_unit_num is not None:
            voter["abode_unit"] = custom_unit_num

        # Add the voter info to the database
        db.set_dict(voter_id, dict(voter), VOTER_DB_IDX)

        # Add voter to this campaign
        db.add_to_set(CAMPAIGN_ID, voter_id, CAMPAIGN_SUBSET_DB_IDX)

        # Add the voter to the abode
        db.set_dict(abode_id, abode, ABODE_DB_IDX)

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
            block_id, abode, custom_unit_num = result

            added: bool = add_voter(
                entry,
                abode,
                custom_unit_num if custom_unit_num != "" else None,
            )

            if not added:
                continue

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


if __name__ == "__main__":
    turnouts: dict[str, float] = json.load(open(turnout_predictions_file))

    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("Usage: python3 associate.py <universe_file>")
        sys.exit(1)

    handle_universe_file(sys.argv[1], turnouts)
