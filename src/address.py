import json
from dataclasses import dataclass

from src.config import street_suffixes_file

street_suffixes: dict[str, str] = json.load(open(street_suffixes_file))


@dataclass(frozen=True)
class Address:
    """
    Represent a standardized address structure.

    The Address class provides a clear representation of various parts of an address.
    It is designed to handle raw address data and offers utility methods related to address manipulation.

    Attributes
    ----------
        house_number (str | None): The primary number for a residential or business property.
        house_number_suffix (str | None): Any suffix to the primary house number (e.g., A, B).
        street_name (str | None): The name of the street or road.
        unit_num (str | None): The apartment or suite number if applicable.
        city (str | None): The name of the city or municipality.
        state (str | None): The state abbreviation or name.
        zip_code (str | None): The postal code.

    Notes
    -----
        This structure aims to bridge the gap between two formats:
        1. addr_num_prefix, addr_num, addr_num_suffix, st_premodifier, st_prefix, st_pretype,
           st_name, st_type, st_postmodifier, unit_type, unit, floor, municipality, county, state, zip_code.
        2. House Number, House Number Suffix, Street Name, Apartment Number, Address Line 2, City, State, Zip from the PA voter file.

    Methods
    -------
        sanitize_street_name: A utility method to standardize the suffix of a given street name.
    """

    # address_pts: addr_num_prefix,addr_num,addr_num_suffix,st_premodifier,st_prefix,st_pretype,st_name,st_type,st_postmodifier
    #              unit_type,unit,floor,municipality,county,state,zip_code
    # universe (PA voter file): House Number,House Number Suffix,Street Name,Apartment Number,Address Line 2,City,State,Zip
    house_number: str | None = None
    house_number_suffix: str | None = None
    street_name: str | None = None
    unit_num: str | None = None
    city: str | None = None
    state: str | None = None
    zip_code: str | None = None

    @staticmethod
    def sanitize_street_name(street_name: str) -> str:
        # Split the street name using spaces as the separator
        words = street_name.casefold().split()

        if len(words) > 1:
            last_word = words[-1]

            # Check if the last word is in the lookup dictionary
            if last_word in street_suffixes:
                # If it is, replace it
                words[-1] = street_suffixes[last_word]

        # Join the words back together and return
        return " ".join(words).rstrip()

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Address):
            return False

        return (
            self.house_number == __value.house_number
            and self.house_number_suffix == __value.house_number_suffix
            and self.street_name == __value.street_name
            and self.unit_num == __value.unit_num
            and self.city == __value.city
            and self.state == __value.state
            and self.zip_code == __value.zip_code
        )


addresses_file_t = dict[Address, tuple[str, str]]
