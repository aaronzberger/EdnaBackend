'''Upload a walk list and its house images to the bucket'''

# TODO: Instead of requesting by coordinates, should be requested by address
# But currently, no more than number and street are stored, so lists need to have more info

# TODO: Perhaps request the address, confirm it is within a distance of desired, and if
# not, request the coordinates

import os
import sys
import json
from decimal import Decimal
import requests
from termcolor import colored
import google_streetview.api
import qrcode
import tqdm
import cv2


API_KEY = os.environ['GOOGLE_API_KEY']
API_SECRET = os.environ['GOOGLE_API_SECRET']


# region Input
if len(sys.argv) != 2:
    print('Usage: python3 submit_list.py <list_file>')
    sys.exit(1)

list_file = sys.argv[1]
if not os.path.isfile(list_file):
    print('File not found: {}'.format(list_file))
    sys.exit(1)
list_file_name = os.path.basename(list_file).split('.')[0]

walk_list = None
with open(list_file, 'r') as f:
    walk_list = json.load(f)

if walk_list is None:
    print('Error: Failed to load walk list')
    sys.exit(1)
# endregion

# region Retrieve house keys
addresses = []
coordinates = []
for block in walk_list['blocks']:
    for house in block['houses']:
        addresses.append(house['address'])
        coordinates.append(house['coordinates'])

# Keys
keys = []
for i in range(len(addresses)):
    lat = Decimal(coordinates[i]['lat']).quantize(Decimal('0.0001'))
    lon = Decimal(coordinates[i]['lon']).quantize(Decimal('0.0001'))
    split_address = addresses[i].split(' ')
    house_num, street = split_address[0], split_address[1]
    keys.append('{}_{}_{}_{}'.format(house_num, street.upper(), lat, lon))
# endregion

# region Check which files don't already exist
needed_images: list[tuple[str, dict]] = []
for key, coords in tqdm.tqdm(zip(keys, coordinates), desc='Checking images', total=len(keys), unit='images', colour='green'):
    url = 'https://votefalcon.s3.amazonaws.com/{}.jpg'.format(key)
    r = requests.head(url)
    if r.status_code != 200:
        needed_images.append((key, coords))

print('Found {}/{} images already. Requesting {} new...'.format(len(keys) - len(needed_images), len(keys), len(needed_images)))
# endregion

print(colored('Confirm requesting {} houses from street view? (y/n)'.format(len(needed_images)), 'yellow'))
if input().lower() != 'y':
    sys.exit(1)

# region Retrieve new images
if os.path.isdir(list_file_name):
    print('Error: Directory already exists: {}'.format(list_file_name))
    sys.exit(1)
os.mkdir(list_file_name)

for image in tqdm.tqdm(needed_images, desc='Downloading images', total=len(needed_images), unit='images', colour='green'):
    params = [{
        'size': '640x640',
        'location': '{},{}'.format(image[1]['lat'], image[1]['lon']),
        'pitch': '10',  # Pitch up slightly to see house
        'fov': '70',  # Fov in slightly to see house
        'key': API_KEY
    }]

    results = google_streetview.api.results(params)

    # Download image
    results.download_links(list_file_name)

    # Rename the image to the key
    os.rename('{}/gsv_0.jpg'.format(list_file_name), '{}/{}.jpg'.format(list_file_name, image[0]))

    # Crop the image a bit (remove 10 pixels from bottom)
    img = cv2.imread('{}/{}.jpg'.format(list_file_name, image[0]))
    img = img[:-25, :]
    cv2.imwrite('{}/{}.jpg'.format(list_file_name, image[0]), img)
# endregion

print(colored('Retrieved and wrote {} images to {}'.format(len(needed_images), list_file_name), 'green'))

# region Generate QR code and display results
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)
qr.add_data(sys.argv[1])
qr.make(fit=True)

img = qr.make_image(fill_color="black", back_color="white")

img.save('{}/QR.png'.format(list_file_name))

# endregion

print(colored('QR code written to {}. Now, upload house images to S3'.format(list_file_name), 'green'))
