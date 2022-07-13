import requests
import folium
import polyline


def get_distance(start_lon: float, start_lat: float, end_lon: float, end_lat: float):
    '''Get the distance on foot (in meters) between two points'''
    loc = '{},{};{},{}'.format(start_lon, start_lat, end_lon, end_lat)
    url = 'http://0.0.0.0:5000/route/v1/walking/' + loc
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()['routes'][0]['distance']


def get_distance_p(p1, p2):
    '''Get the distance on foot (in meters) between two points'''
    return get_distance(p1.lon, p1.lat, p2.lon, p2.lat)


def get_route(start_lon, start_lat, end_lon, end_lat):
    loc = '{},{};{},{}'.format(start_lon, start_lat, end_lon, end_lat)
    url = 'http://0.0.0.0:5000/route/v1/walking/' + loc
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


def get_map(route):
    m = folium.Map(location=[(route['start_point'][0] + route['end_point'][0]) / 2,
                             (route['start_point'][1] + route['end_point'][1]) / 2],
                   zoom_start=13)

    folium.PolyLine(
        route['route'],
        weight=8,
        color='blue',
        opacity=0.6
    ).add_to(m)

    folium.Marker(
        location=route['start_point'],
        icon=folium.Icon(icon='play', color='green')
    ).add_to(m)

    folium.Marker(
        location=route['end_point'],
        icon=folium.Icon(icon='stop', color='red')
    ).add_to(m)

    return m


map = get_map(get_route(-79.932296, 40.442603, -79.931921, 40.446955))
map.save('test_route.html')
