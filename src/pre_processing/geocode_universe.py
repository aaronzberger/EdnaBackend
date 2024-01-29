"""Associate PA voter file addresses with the geographic data, determine voter and abode values,
and save the results to be used in the optimizer.

A universe file is provided, for which addresses are from the PA voter export.

These addresses are mapped to longitude and latitude via SmartyStreets API
"""
import os

from smartystreets_python_sdk import SharedCredentials, StaticCredentials, exceptions, Batch, ClientBuilder
from smartystreets_python_sdk.us_street import Lookup as StreetLookup
from smartystreets_python_sdk.us_street.match_type import MatchType
import pandas as pd
import numpy as np
from src.config import universe_file, geocoded_universe_file

def load_universe(filename):
  universe = pd.read_csv(filename)
  return universe.head(5)
def query_smarty(universe):

  auth_id = '06b04f81-fd8c-9bda-b876-2b1a05c57b07'
  auth_token = 'sE3Koj62HjEvBQ6zUum7'
  credentials = StaticCredentials(auth_id, auth_token)
  # The appropriate license values to be used for your subscriptions
  # can be found on the Subscriptions page of the account dashboard.
  # https://www.smartystreets.com/docs/cloud/licensing
  # client = ClientBuilder(credentials).with_licenses(["us-core-cloud"]).build_us_street_api_client()
  client = ClientBuilder(credentials).with_custom_header({'User-Agent': 'smartystreets (python@0.0.0)', 'Content-Type': 'application/json'}).build_us_street_api_client()
  # client = ClientBuilder(credentials).with_proxy('localhost:8080', 'user', 'password').build_us_street_api_client()
  # Uncomment the line above to try it with a proxy instead

  # Documentation for input fields can be found at:
  # https://smartystreets.com/docs/us-street-api#input-fields

  batch = Batch()

  # Documentation for input fields can be found at:
  # https://smartystreets.com/docs/us-street-api#input-fields

  for i, row in universe.iterrows():
     
    batch.add(StreetLookup())
    batch[i].street = str(row["House Number"]) + " " + str(row["House Number Suffix"]) + " " + row["Street Name"]
    # batch[i].street2 = row["Address Line 2"]
    batch[i].secondary = str(row["Apartment Number"])
    batch[i].city = row["City"]
    batch[i].state = row["State"]
    batch[i].zipcode = str(row["Zip"])
    batch[i].lastline = row["City"] + ", " + row["State"] + " " + str(row["Zip"])
    batch[i].addressee = row["First Name"] + " " + row["Middle Name"] + " " + row["Last Name"]
    batch[i].candidates = 1
    batch[i].match = MatchType.ENHANCED  # "invalid" is the most permissive match,
                                        # this will always return at least one result even if the address is invalid.
                                        # Refer to the documentation for additional Match Strategy options.l
  assert len(batch) == 5

  try:
    client.send_batch(batch)
  except exceptions.SmartyException as err:
    print
    print(err)
    return
  longs = []
  lats = []
  for i, lookup in enumerate(batch):
    candidates = lookup.result

    if len(candidates) == 0:
      longs.append(np.nan)
      lats.append(np.nan)
      continue

    metadata = candidates[0].metadata
    longs.append(metadata.longitude)
    lats.append(metadata.latitude)
  universe.insert(1, "longitude", longs, True)
  universe.insert(1, "latitude", lats, True)
  return universe

if __name__ == "__main__":
  universe = load_universe(universe_file)
  universe = query_smarty(universe)
  universe.to_csv(geocoded_universe_file, index=False)
  

