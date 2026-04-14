import csv
from pathlib import Path

missing = 0
total = 0

with open("combined.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        total += 1
        if not Path(row["image"]).exists():
            missing += 1

print("total:", total)
print("missing:", missing)