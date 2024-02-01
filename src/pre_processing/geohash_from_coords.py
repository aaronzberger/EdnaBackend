import geohash
import numpy as np
import pandas as pd
from src.config import geocoded_universe_file, GEOHASH_PRECISION
universe = pd.read_csv(geocoded_universe_file).head(5)
lat_long = list(zip(universe['latitude'], universe['longitude']))
geohashes = np.array([geohash.encode(i[0], i[1], precision = GEOHASH_PRECISION) for i in lat_long])
#next step: check which geohashes are in our hash table database, call make_blocks workflow on those that aren't.
