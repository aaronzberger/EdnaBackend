import os
import pandas as pd

from src.config import BASE_DIR


NAME = "mail_data_10-27-23"


# Read the data from the file and take first row as header
data = pd.read_csv(
    os.path.join(BASE_DIR, "input", f"{NAME}.txt"),
    sep="\t",
    header=0,
    encoding="unicode_escape",
)

# Print any duplicate ID Numbers
if len(data[data.duplicated("ID Number")]["ID Number"]) > 0:
    print("Duplicate ID Numbers:")
    print("\t", data[data.duplicated("ID Number")]["ID Number"])

# Make ID Number the index
data.set_index("ID Number", inplace=True)

# Count the number of entries with BallotSent not null
print("BallotSent not null:", len(data[data["BallotSent"].notnull()]))
print("BallotReturned not null:", len(data[data["BallotReturned"].notnull()]))
print("VoteRecorded not null:", len(data[data["VoteRecorded"].notnull()]))

# Keep only some columns
data = data[
    [
        "Election",
        "AppType",
        "FullName",
        "PrecinctSplit",
        "PartyDesc",
        "AppIssuedDate",
        "AppReturnedDate",
        "BallotSent",
        "BallotReturned",
        "VoteRecorded",
        "BallotStyle",
    ]
]

# Write to json
data.to_json(os.path.join(BASE_DIR, "input", f"{NAME}.json"), orient="index")
