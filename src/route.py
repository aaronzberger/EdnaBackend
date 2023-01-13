from typing import Any

import polyline
import requests
from src.config import Point

# SERVER = 'http://0.0.0.0:5000'
SERVER = 'http://172.17.0.3:5000'


def get_distance(start: Point, end: Point) -> float:
    '''
    Get the distance on foot (in meters) between two points

    Parameters:
        start (Point): the starting point
        end (Point): the ending point

    Returns:
        float: the distance (in meters), on foot, to walk from start to end
    '''
    loc = '{},{};{},{}'.format(start['lon'], start['lat'], end['lon'], end['lat'])
    url = SERVER + '/route/v1/walking/' + loc
    try:
        r = requests.get(url)
    except Exception:
        raise RuntimeError('Request to OSRM server failed. Is it running?')
    if r.status_code == 200:
        return r.json()['routes'][0]['distance']
    raise RuntimeError('Could not contact OSRM server')


def get_route(start: Point, end: Point) -> dict[str, Any]:
    '''
    Get the full route on foot between two points

    Parameters:
        start (Point): the starting point
        end (Point): the ending point

    Returns:
        dict:
            'route' (list): the route, as given by polyline
            'start_point' (list): the starting point in the format [lat, lon]
            'end_point' (list): the ending point in the format [lat, lon]
            'distance' (float): the distance from start to end
    '''
    loc = '{},{};{},{}'.format(start['lon'], start['lat'], end['lon'], end['lat'])
    url = SERVER + '/route/v1/walking/' + loc
    r = requests.get(url)
    if r.status_code != 200:
        return {}

    res = r.json()
    routes = polyline.decode(res['routes'][0]['geometry'])
    start_point = [res['waypoints'][0]['location'][1], res['waypoints'][0]['location'][0]]
    end_point = [res['waypoints'][1]['location'][1], res['waypoints'][1]['location'][0]]
    distance = res['routes'][0]['distance']

    return {
        'route': routes,
        'start_point': start_point,
        'end_point': end_point,
        'distance': distance
    }
