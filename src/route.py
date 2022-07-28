import polyline
import requests

from src.gps_utils import Point

SERVER = 'http://0.0.0.0:5000'


def get_distance(p1: Point, p2: Point) -> float:
    '''Get the distance on foot (in meters) between two points'''
    loc = '{},{};{},{}'.format(p1.lon, p1.lat, p2.lon, p2.lat)
    url = SERVER + '/route/v1/walking/' + loc
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()['routes'][0]['distance']
    raise RuntimeError('Could not contact OSRM server')


def get_route(start: Point, end: Point) -> dict:
    loc = '{},{};{},{}'.format(start.lon, start.lat, end.lon, end.lat)
    url = SERVER + '/route/v1/walking/' + loc
    r = requests.get(url)
    if r.status_code != 200:
        return {}

    res = r.json()
    routes = polyline.decode(res['routes'][0]['geometry'])
    start_point = [res['waypoints'][0]['location'][1], res['waypoints'][0]['location'][0]]
    end_point = [res['waypoints'][1]['location'][1], res['waypoints'][1]['location'][0]]
    distance = res['routes'][0]['distance']

    out = {'route': routes,
           'start_point': start_point,
           'end_point': end_point,
           'distance': distance
           }

    return out
