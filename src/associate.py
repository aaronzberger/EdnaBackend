import dataclasses
import pprint
from rapidfuzz import fuzz, process
from src.utils.address import Address, addresses_file_t
import jsonpickle
from src.config import addresses_file, street_suffixes_file, manual_match_output_file
import json


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
