# movie_sentiment_auto.py
# Simple Movie Review Sentiment Analyzer (auto-download via kagglehub)
# Requires: pip install scikit-learn pandas kagglehub

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import kagglehub
import os

# ---------- STEP 1: DOWNLOAD DATA ----------
print("[+] Downloading IMDB dataset from Kaggle...")
dataset_path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
csv_path = os.path.join(dataset_path, "IMDB Dataset.csv")

if not os.path.exists(csv_path):
    print(f"[!] Dataset CSV not found in {dataset_path}")
    exit()

print("[+] Loading dataset...")
df = pd.read_csv(csv_path)

# ---------- STEP 2: PREPARE DATA ----------
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.2, random_state=42
)

# ---------- STEP 3: VECTORIZE TEXT ----------
print("[+] Converting text to vectors...")
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------- STEP 4: TRAIN MODEL ----------
print("[+] Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# ---------- STEP 5: EVALUATE ----------
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"[+] Model accuracy: {accuracy * 100:.2f}%")

# ---------- STEP 6: INTERACTIVE TEST ----------
print("\nType a movie review to test the sentiment (or type 'quit' to exit):")
while True:
    text = input("> ")
    if text.lower() == "quit":
        break
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    print(f"Prediction: {prediction}")
