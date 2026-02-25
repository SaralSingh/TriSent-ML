from pathlib import Path
import shutil
import subprocess

DATA_DIR = Path("data")

MAIN_DATA = DATA_DIR / "sentiment.txt"
FEEDBACK = DATA_DIR / "feedback.txt"

if not FEEDBACK.exists():
    print("No feedback to retrain.")
    exit()

print("Merging feedback into training data...")

with open(MAIN_DATA, "a", encoding="utf-8") as main, \
     open(FEEDBACK, "r", encoding="utf-8") as fb:
    main.write("\n")
    main.write(fb.read())

FEEDBACK.unlink()  # clear feedback

print("Retraining model...")
subprocess.run(["python", "train.py"])

print("✅ Retraining complete.")