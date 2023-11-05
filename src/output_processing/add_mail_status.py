"""
Augment every walk list file with additional attributes per voter:
    - BallotSent
    - BallotReturned
"""

from src.config import mail_data_file, files_dir
import json
import os

mail_data = json.load(open(mail_data_file))

output_dir = os.path.join(files_dir, "mail_augmented")
os.makedirs(output_dir, exist_ok=True)

voters_found = 0
voters_not_found = 0

for file in os.listdir(files_dir):
    if not file.endswith(".json"):
        continue

    data = json.load(open(os.path.join(files_dir, file)))
    for block in data["blocks"]:
        for house in block["houses"]:
            for voter in house["voter_info"]:
                voter_id = voter["voter_id"]

                # Lookup voter in mail data
                if voter_id in mail_data:
                    voter["ballot_sent"] = mail_data[voter_id]["BallotSent"]
                    voter["ballot_returned"] = mail_data[voter_id]["BallotReturned"]
                    voter["app_returned"] = mail_data[voter_id]["AppReturnedDate"]
                    voters_found += 1
                else:
                    voter["ballot_sent"] = None
                    voter["ballot_returned"] = None
                    voter["app_returned"] = None
                    voters_not_found += 1

    json.dump(data, open(os.path.join(output_dir, file), "w"), indent=4)

print(f"Voters found: {voters_found}")
print(f"Voters not found: {voters_not_found}")
