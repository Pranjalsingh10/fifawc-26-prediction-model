import csv

with open(
    "data/raw/SquadLists-English.csv",
    encoding="utf-8",
    errors="ignore"
) as f:

    reader = csv.reader(f)

    for i, row in enumerate(reader):

        print(i, row)

        if i > 50:
            break