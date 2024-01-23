"""Pull uploaded data from S3, process it, and make a mailer for un-hit houses"""

import csv
import os
import sys
import json
import boto3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.config import BASE_DIR, InternalPoint, details_file, files_dir, house_to_voters_file
from src.utils.viz import display_visited_and_unvisited


# Connect to S3
s3 = boto3.client('s3', aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'), aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))


VISTA_FIELDNAMES = ["Salutation", "First name", "Middle", "Name", "Suffix", "Title",
                    "Company (Required if no Last name)", "Address Line 1 (Required)", "Address Line 2",
                    "City (Required)", "State (Required)", "Zip Code (Required. 5- or 9- digits)"]


house_to_voters = json.load(open(house_to_voters_file))


output_file = open(os.path.join(BASE_DIR, 'all_missed.csv'), 'w')
masonic_file = open(os.path.join(BASE_DIR, 'masonic.csv'), 'w')
writer = csv.DictWriter(output_file, fieldnames=VISTA_FIELDNAMES)
writer.writeheader()

masonic_writer = csv.DictWriter(masonic_file, fieldnames=VISTA_FIELDNAMES)
masonic_writer.writeheader()


# List all files in the S3 bucket
def list_files(bucket_name) -> set[str]:
    """List all files in the S3 bucket"""
    response = s3.list_objects_v2(Bucket=bucket_name)

    filenames = [file['Key'] for file in response['Contents']]

    filename_set = set(filenames)
    assert len(filename_set) == len(filenames), 'Duplicate filenames in S3 bucket'

    return set(filenames)


# Details of the lists
details = json.load(open(details_file))
all_list_files = set(details.keys())

uploaded_files = list_files('votefalcon-uploads')
uploaded_files = {file[:-5] for file in uploaded_files}

# Find which lists have been uploaded
uploaded_lists = all_list_files.intersection(uploaded_files)

uuids_hit: set[str] = set()
uuids_miss: set[str] = set()
uuids_unvisited: set[str] = set()

tooltips = {}

uuids_hit_pts: list[InternalPoint] = []
uuids_miss_pts: list[InternalPoint] = []
uuids_unvisited_pts: list[InternalPoint] = []


for list_name in uploaded_lists:
    print(f'Processing {list_name}')

    filename = list_name + '.json'

    # Retrieve the file from S3 as json
    file_obj = s3.get_object(Bucket='votefalcon-uploads', Key=filename)

    try:
        file_data = json.loads(file_obj['Body'].read().decode('utf-8'))
    except json.decoder.JSONDecodeError:
        print(f'Error decoding {filename}')
        continue

    # Read the corresponding file
    list_file = json.load(open(os.path.join(files_dir, filename)))

    address_to_uuid = {}
    for block in list_file["blocks"]:
        for house in block["abodes"]:
            if house['uuid'] in address_to_uuid:
                print(f'Duplicate UUID: {house["uuid"]} with address {house["display_address"]} from list {list_name}')
            else:
                address_to_uuid[house["display_address"]] = house["uuid"]

    assert list_file["id"] == list_name

    # Read all houses from the file
    for block in file_data["blocks"]:
        for house in block["abodes"]:
            uuid = address_to_uuid[house["address"]]

            # Remove the address from the dict
            del address_to_uuid[house["address"]]

            voter_info = house_to_voters[uuid]

            tooltips[uuid] = voter_info['display_address']

            home_question_reached = False

            for question in house['voter_info']:
                if question['question'] == 'Was the voter home?':
                    home_question_reached = True
                    if question['answer'] == 'Yes':
                        if uuid in uuids_hit.union(uuids_miss).union(uuids_unvisited):
                            print(f'(hit) Duplicate UUID: {uuid} with address {house["address"]} from list {list_name}')
                        uuids_hit.add(uuid)
                        uuids_hit_pts.append(InternalPoint(lat=voter_info['latitude'], lon=voter_info['longitude'], id=uuid))
                    elif question['answer'] == 'No':
                        if uuid in uuids_hit.union(uuids_miss).union(uuids_unvisited):
                            print(f'(miss) Duplicate UUID: {uuid} with address {house["address"]} from list {list_name}')
                        uuids_miss.add(uuid)
                        uuids_miss_pts.append(InternalPoint(lat=voter_info['latitude'], lon=voter_info['longitude'], id=uuid))
                    else:
                        raise ValueError(f'Invalid answer: {question["answer"]}')
                else:
                    tooltips[uuid] += f'<br>{question["question"]}: <b>{question["answer"]}</b>'

            if not home_question_reached:
                uuids_miss.add(uuid)

    # Add all houses in the list but not in the uploaded file
    for uuid in address_to_uuid.values():
        if uuid in uuids_hit.union(uuids_miss).union(uuids_unvisited):
            print(f'(unvisited) Duplicate UUID: {uuid} with address {house["address"]} from list {list_name}')

        uuids_unvisited.add(uuid)
        tooltips[uuid] = house_to_voters[uuid]['display_address']
        uuids_unvisited_pts.append(InternalPoint(lat=house_to_voters[uuid]['latitude'], lon=house_to_voters[uuid]['longitude'], id=uuid))

# Write a list of all houses that were not visited
for uuid in uuids_unvisited.union(uuids_miss):
    # Lookup the voters in this house
    voter_info = house_to_voters[uuid]

    uuids_unvisited_pts.append(InternalPoint(lat=voter_info['latitude'], lon=voter_info['longitude'], id=uuid))

    names = [v['name'] for v in voter_info['voter_info']]

    # If there are more than 2 voters, find a common last name and use that
    if len(names) > 2:
        last_names = [name.split(' ')[-1] for name in names]
        last_name = max(set(last_names), key=last_names.count)
        display_name = f'{last_name} FAMILY'
    else:
        display_name = ' & '.join(names)

    writer.writerow({
        'Salutation': '',
        'First name': '',
        'Middle': '',
        'Name': display_name,
        'Suffix': '',
        'Title': '',
        'Company (Required if no Last name)': '',
        'Address Line 1 (Required)': voter_info['display_address'],
        'Address Line 2': '',
        'City (Required)': voter_info['city'],
        'State (Required)': voter_info['state'],
        'Zip Code (Required. 5- or 9- digits)': voter_info['zip']
    })


# Also read in 29-0
masonic_file = list_file = json.load(open(os.path.join(files_dir, 'rosselli-29-0.json')))

for block in masonic_file["blocks"]:
    for house in block["abodes"]:
        uuid = house["uuid"]

        voter_info = house_to_voters[uuid]

        tooltips[uuid] = voter_info['display_address']

        names = [v['name'] for v in voter_info['voter_info']]

        uuids_unvisited_pts.append(InternalPoint(lat=voter_info['latitude'], lon=voter_info['longitude'], id=uuid))

        # If there are more than 2 voters, find a common last name and use that
        if len(names) > 2:
            last_names = [name.split(' ')[-1] for name in names]
            last_name = max(set(last_names), key=last_names.count)
            display_name = f'{last_name} FAMILY'
        else:
            display_name = ' & '.join(names)

        writer.writerow({
            'Salutation': '',
            'First name': '',
            'Middle': '',
            'Name': display_name,
            'Suffix': '',
            'Title': '',
            'Company (Required if no Last name)': '',
            'Address Line 1 (Required)': voter_info['display_address'],
            'Address Line 2': '',
            'City (Required)': voter_info['city'],
            'State (Required)': voter_info['state'],
            'Zip Code (Required. 5- or 9- digits)': voter_info['zip']
        })

        masonic_writer.writerow({
            'Salutation': '',
            'First name': '',
            'Middle': '',
            'Name': display_name,
            'Suffix': '',
            'Title': '',
            'Company (Required if no Last name)': '',
            'Address Line 1 (Required)': voter_info['display_address'],
            'Address Line 2': '',
            'City (Required)': voter_info['city'],
            'State (Required)': voter_info['state'],
            'Zip Code (Required. 5- or 9- digits)': voter_info['zip']
        })


print(f"""
    Total number of voters: {len(uuids_hit) + len(uuids_miss) + len(uuids_unvisited)}
    Number of voters hit: {len(uuids_hit)}
    Number of voters missed: {len(uuids_miss)}
    Number of voters unvisited: {len(uuids_unvisited)}
""")


display_visited_and_unvisited(list(uuids_hit_pts), list(uuids_miss_pts + uuids_unvisited_pts), tooltips).save(os.path.join(BASE_DIR, "viz", "visited_and_unvisited.html"))
