import csv
import os
import sys

from geographiclib.geodesic import Geodesic

EARTH_R = 6371e3;  # meters

converter = Geodesic.WGS84

if len(sys.argv) != 2 or not os.path.exists(sys.argv[1]):
    print('Usage: distances.py [CSV FILE PATH]')

write_file = '{}_results.csv'.format(sys.argv[1])

first_point = None

csv_writer = csv.writer(open(write_file, 'w'))

for line in csv.reader(open(sys.argv[1], 'r')):
    if first_point is None:
        first_point = line[1:]
        first_point[0] = float(first_point[0])
        first_point[1] = float(first_point[1])
    if line[1] == '' or line[2] == '':
        csv_writer.writerow([line[0]])
        continue
    distance = converter.Inverse(
        lat1=first_point[0], lon1=first_point[1],
        lat2=float(line[1]), lon2=float(line[2]))['s12']
    csv_writer.writerow([line[0], distance])

print('Wrote results to {}'.format(write_file))