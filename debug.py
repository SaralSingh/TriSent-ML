from pathlib import Path

path = Path("data/sentiment.txt")

count = 0
with open(path, encoding="utf-8", errors="ignore") as f:
    for line in f:
        if "\t" in line:
            count += 1
        if count <= 5:
            print(line.strip())

print("VALID LINES:", count)