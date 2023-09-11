"""Upload a walk list and its house images to the bucket."""

# TODO: Instead of requesting by coordinates, should be requested by address
# But currently, no more than number and street are stored, so lists need to have more info

# TODO: Perhaps request the address, confirm it is within a distance of desired, and if
# not, request the coordinates

import os
import sys
import json
import requests
from termcolor import colored
import google_streetview.api
import qrcode
import tqdm
import cv2
import argparse

from src.config import street_view_failed_uuids_file


arg_parser = argparse.ArgumentParser(description='Retrieve new house images from Street View')
arg_parser.add_argument('-d', '--directory', type=str, help='Directory of lists', required=True)
arg_parser.add_argument('-k', '--key', type=str, help='Google API key', default=os.environ['GOOGLE_API_KEY'])
arg_parser.add_argument('-a', '--approve', action='store_true', help='Automatically approve downloading images')
args = arg_parser.parse_args()


failed_uuids = json.load(open(street_view_failed_uuids_file))


def process_list(filename: str, save_dir: str):
    walk_list = json.load(open(filename))

    # Retrieve house keys
    addresses: list[str] = []
    uuids: list[str] = []
    for block in walk_list['blocks']:
        for house in block['houses']:
            addresses.append(f"{house['display_address']}, {house['city']}, {house['state']}, {house['zip']}".replace('None', ''))
            uuids.append(house["uuid"])

    num_failed = num_found = num_to_upload = num_needed = 0

    # Check which files don't already exist
    needed_images: list[tuple[str, str]] = []
    for uuid, address in tqdm.tqdm(zip(uuids, addresses), desc='Checking images', total=len(uuids), unit='images', colour='green'):
        if uuid in failed_uuids:
            num_failed += 1
            continue

        url = 'https://votefalcon.s3.amazonaws.com/images/{}.jpg'.format(uuid)
        r = requests.head(url)
        if r.status_code != 200:
            # Check if we have it locally but not uploaded
            if os.path.isfile('{}/{}.jpg'.format(save_dir, uuid)):
                # Move it to the upload directory
                os.rename('{}/{}.jpg'.format(save_dir, uuid), '{}/{}/{}.jpg'.format(save_dir, 'upload', uuid))
                num_to_upload += 1
                continue

            needed_images.append((uuid, address))
            num_needed += 1
        else:
            num_found += 1

    print(f"Of {len(uuids)} images, {num_found} already exist, {num_failed} failed, {num_to_upload} need to be uploaded, and {num_needed} need to be requested")

    if len(needed_images) == 0:
        return

    # Ensure user approval for requesting images
    if not args.approve:
        print(colored('Confirm requesting {} houses from street view? (y/n)'.format(len(needed_images)), 'yellow'))
        if input().lower() != 'y':
            sys.exit(1)

    for image in tqdm.tqdm(needed_images, desc='Downloading images', total=len(needed_images), unit='images', colour='green'):

        params = [{
            'size': '640x640',
            'location': image[1],
            'key': args.key,
        }]

        results = google_streetview.api.results(params)

        # Download image to the house_images directory
        results.download_links('{}/'.format(save_dir))
        results.save_metadata('{}/metadata.json'.format(save_dir))

        if not os.path.isfile('{}/gsv_0.jpg'.format(save_dir)):
            print('Error: Failed to download image for {}: {}'.format(image[0], image[1]))
            failed_uuids[image[0]] = False
            json.dump(failed_uuids, open(street_view_failed_uuids_file, 'w'))
            continue

        # Rename the image to the key
        os.rename('{}/gsv_0.jpg'.format(save_dir), '{}/{}.jpg'.format(save_dir, image[0]))

        # Crop the image a bit (remove 10 pixels from bottom)
        img = cv2.imread('{}/{}.jpg'.format(save_dir, image[0]))
        img = img[:-25, :]
        cv2.imwrite('{}/{}.jpg'.format(save_dir, image[0]), img)

    print(colored('Retrieved and wrote {} images to {}'.format(len(needed_images), save_dir), 'green'))


if not os.path.isdir(args.directory):
    print(colored('Error: {} is not a directory'.format(args.directory), 'red'))
    sys.exit(1)

# Get all lists
lists = [f for f in os.listdir(args.directory) if os.path.isfile(os.path.join(args.directory, f)) and f.endswith('.json')]
if len(lists) == 0:
    print(colored('Error: No lists found in {}'.format(args.directory), 'red'))
    sys.exit(1)

image_save_dir = os.path.join(args.directory, 'house_images')
qr_save_dir = os.path.join(args.directory, 'qr_codes')

for list_file_name in lists:
    print(colored('Processing {}'.format(list_file_name), 'yellow'))

    # Get the list ID
    list_id = list_file_name.split('.')[0]

    # Download the images
    process_list(os.path.join(args.directory, list_file_name), image_save_dir)

    # Generate QR code and display results
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data("https://votefalcon.s3.amazonaws.com/lists/{}.json".format(list_file_name))
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    img.save(os.path.join(qr_save_dir, '{}.png'.format(list_id)))
