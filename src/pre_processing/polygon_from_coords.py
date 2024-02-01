from src.config import geocoded_universe_file
import pandas as pd
import numpy as np
from pyproj import Proj
from scipy.spatial import ConvexHull
from shapely import Polygon
from shapely import affinity


universe = pd.read_csv(geocoded_universe_file).head(5)
p_merc = Proj(proj='merc')
longitudes = universe['longitude'].to_numpy()
latitudes = universe['latitude'].to_numpy()
transformed = np.transpose(np.asarray(p_merc(longitudes, latitudes)))
hull = ConvexHull(transformed)
hull_points = np.array([transformed[i] for i in hull.vertices])
p = Polygon(hull_points)
scaled_polygon = affinity.scale(p, xfact=1.2, yfact=1.2)
xx, yy = scaled_polygon.exterior.coords.xy
polygon = np.transpose(np.asarray(p_merc(np.array(xx), np.array(yy), inverse=True)))
#for some reason the inverse projection puts latitude before longitude, so you have to swap it back
polygon[:, [0, 1]] = polygon[:, [1, 0]]p
output_string = ""
for point in polygon:
  output_string += str(point[0]) + " " + str(point[1]) + " "
output_string = output_string.strip()


