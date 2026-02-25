import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

input_file = BASE_DIR / "generic_sentiment_dataset_50k.csv"   # your actual CSV
output_file = BASE_DIR / "trisent.txt"

written = 0

with open(input_file, newline='', encoding="utf-8", errors="ignore") as csvfile, \
     open(output_file, "w", encoding="utf-8") as outfile:

    reader = csv.reader(csvfile)
    header = next(reader, None)  # skip header

    for row in reader:
        if len(row) < 3:
            continue

        text = row[1].strip().lower()
        label = row[2].strip()

        if not text:
            continue

        # numeric label → TriSent
        if label == "0":
            sentiment = "negative"
        elif label == "1":
            sentiment = "mixed"
        elif label == "2":
            sentiment = "positive"
        else:
            continue

        outfile.write(f"{sentiment}|{text}\n")
        written += 1

print("Written samples:", written)