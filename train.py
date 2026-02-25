import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from pathlib import Path

DATA_PATH = Path("data/sentiment.txt")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

print("Loading dataset...")

X = []
y = []

with open(DATA_PATH, encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # split ONLY on first |
        if "|" not in line:
            continue

        label, text = line.split("|", 1)

        label = label.strip()
        text = text.strip()

        if not label or not text:
            continue

        X.append(text)
        y.append(label)

print(f"Loaded samples: {len(X)}")

if len(X) == 0:
    raise RuntimeError("❌ No valid training samples found")

print("Training model...")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000,
    stop_words="english"
)

X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(
    max_iter=500,
    n_jobs=-1
)

model.fit(X_vec, y)

joblib.dump(vectorizer, MODEL_DIR / "vectorizer.joblib")
joblib.dump(model, MODEL_DIR / "sentiment_model.joblib")

print("✅ Training complete. Model saved.")