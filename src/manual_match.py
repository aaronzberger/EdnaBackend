import json
import os
import googlemaps
from termcolor import cprint
import termcolor
from src.config import InternalPoint
from src.utils.gps import great_circle_distance

from src.config import manual_match_input_file, manual_match_output_file, reverse_geocode_file, id_to_addresses_file, GOOGLE_MAPS_API_KEY
gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)


def load_id_to_address_mapping():
    """
    Load UUID to address mapping from the file.
    """
    with open(id_to_addresses_file, 'r') as file:
        return json.load(file)

def get_coordinates_from_google(address_str):
    """
    Fetch coordinates (latitude, longitude) for an address using Google Maps Geocoding API.
    """
    cprint("Making google maps api call", "green")

    geocode_result = gmaps.geocode(address_str)
    if geocode_result:
        location = geocode_result[0]['geometry']['location']
        if geocode_result[0]['geometry']['location_type'] != "ROOFTOP":
            print("Geocode request returned low-confidence result")
        return (location['lat'], location['lng'])
    return None


def find_closest_houses(coords: tuple, n=5, limit=100):
    """
    Find the n closest houses to the given coordinates.
    """
    with open(reverse_geocode_file, 'r') as file:
        houses = json.load(file)

    distances = [
        (house, great_circle_distance(
            {'lat': coords[0], 'lon': coords[1], 'type': '', 'id': ''},
            {'lat': house[0], 'lon': house[1], 'type': '', 'id': ''}
        )) for house in houses
    ]
    distances.sort(key=lambda x: x[1])  # Sort by distance
    distances = [house for house in distances if house[1] <= limit]
    return [house[0][2] for house in distances[:n]]


def display_address(address):
    """
    Formats and displays an address dictionary.
    """
    components = [address['house_number'], address['house_number_suffix'], address['street_name'], address['unit_num'], address['zip_code']]
    return " ".join([component for component in components if component != ""])


def load_matches():
    """
    Loads matches from a json file. If the file doesn't exist, return an empty list.
    """
    if os.path.exists(manual_match_output_file):
        with open(manual_match_output_file, 'r') as file:
            return json.load(file)
    return []


def save_matches(matches):
    """
    Saves the list of matched addresses to a json file.
    """

    with open(manual_match_output_file, 'w') as file:
        json.dump(matches, file)


def match_addresses(data):
    """
    Prompts user for matching addresses.
    """
    matched_pairs = load_matches()
    uuid_address_mapping = load_id_to_address_mapping()

    # Filter out addresses that already have a saved match
    unmatched_addresses = [
        entry for entry in data if not any(
            match['universe']['house_number'] == entry['universe']['house_number'] and
            match['universe']['street_name'] == entry['universe']['street_name'] and
            match['universe']['zip_code'] == entry['universe']['zip_code'] for match in matched_pairs)
    ]

    for entry in unmatched_addresses:
        print("\nUniverse Address:")
        cprint(display_address(entry['universe']), "light_blue")

        if not entry['choices']:
            print("No local matching choices available, fetching nearest houses from Google")
            address_str = display_address(entry['universe'])
            coords = get_coordinates_from_google(address_str)
            if coords:
                closest_houses = find_closest_houses(coords)
                entry['choices'].extend([(house, 0.0, 0, "", "") for house in
                                         closest_houses])  # The additional values in the tuple are placeholders

        if entry['choices']:
            print("\nChoices:")
            for index, choice in enumerate(entry['choices']):
                print(f"{index + 1}. {display_address(choice[0])}")
        else:
            print("No automated choices found, good luck")

        while True:
            selection = input("\nEnter choice number, UUID, 'no exist', 'not found', or 'skip': ")

            # If user provides a UUID
            if selection in uuid_address_mapping:
                chosen_match = uuid_address_mapping[selection]
                print(f"Matched with UUID: {display_address(chosen_match)}")

                unit_num = ""
                if entry["universe"]["unit_num"] != "":
                    while True:
                        response = input(
                            f"Do you want to save the unit number from the voter file? ({entry['universe']['unit_num']}) y/n")
                        if response == "y" or response == "Y":
                            unit_num = entry["universe"]["unit_num"]
                # Add the match to the list

                # Add the match to the list
                matched_pairs.append({"universe": entry['universe'], "match": chosen_match, "unit_num": unit_num})

                # Save the matches to the JSON file
                save_matches(matched_pairs)
                break
            elif selection == "skip":
                break
            elif selection == "no exist":
                # Add with an empty match
                matched_pairs.append({"universe": entry['universe'], "match": "DNE"})
                save_matches(matched_pairs)
                break
            elif selection == "not found":
                # Add with a None match
                matched_pairs.append({"universe": entry['universe'], "match": {}})
                save_matches(matched_pairs)
                break
            elif selection.isdigit() and 0 < int(selection) <= len(entry['choices']):
                chosen_match = entry['choices'][int(selection) - 1][0]
                print(f"Matched with: {display_address(chosen_match)}")
                unit_num = ""
                if entry["universe"]["unit_num"] != "":
                    while True:
                        response = input(f"Do you want to save the unit number from the voter file? ({entry['universe']['unit_num']}) y/n: ")
                        if response == "y" or response == "Y":
                            unit_num = entry["universe"]["unit_num"]
                            print("Saving unit number")
                            break
                        else:
                            print("Not saving unit number")
                            break
                # Add the match to the list
                matched_pairs.append({"universe": entry['universe'], "match": chosen_match, "unit_num": unit_num})

                # Save the matches to the JSON file
                save_matches(matched_pairs)
                break
            else:
                print("Invalid input. Try again.")

        print("-----------------------------")


if __name__ == "__main__":
    match_addresses(json.load(open(manual_match_input_file)))
