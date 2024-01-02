import qrcode
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers import RoundedModuleDrawer
from qrcode.image.styles.colormasks import RadialGradiantColorMask
from PIL import Image
import sys
import os
import argparse
from termcolor import colored

from src.config import STYLE_COLOR

parser = argparse.ArgumentParser(description='Create a QR code from a string, file, or directory')
parser.add_argument('input', type=str, help='The input string, file, or directory')
parser.add_argument('output', type=str, help='The output file or directory')
args = parser.parse_args()


def create_qr_code(data):
    qr = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        # version=1,
        border=4,
        box_size=10
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(image_factory=StyledPilImage, color_mask=RadialGradiantColorMask(edge_color=(15, 107, 245)))

    return img


def filename_to_qr_link(filename):
    return f"https://votefalcon.s3.amazonaws.com/dev_1/lists/{filename}"


if os.path.isdir(args.input):
    print(colored('Creating QR codes from directory', 'green'))
    for file in os.listdir(args.input):
        if file.endswith('.json'):
            print(colored('Creating QR code from file ' + file, 'green'))
            img = create_qr_code(filename_to_qr_link(file))
            img.save(args.output + '/' + file[:-5] + '.png')
elif os.path.isfile(args.input):
    print(colored('Creating QR code from file', 'green'))
    img = create_qr_code(filename_to_qr_link(args.input))
    img.save(args.output)
else:
    print(colored('Creating QR code from string', 'green'))
    img = create_qr_code(args.input)
    img.save(args.output)
