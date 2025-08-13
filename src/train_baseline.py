import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import re
from tensorflow.keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense

# --- Text cleaning ---
def clean_text(text):
    text = re.sub(r"<.*?>", "", text)  # remove HTML tags
    text = re.sub(r"[^a-zA-Z ]", "", text)  # keep only letters and spaces
    return text.lower()

# --- Load dataset ---
df = pd.read_csv("IMDB Dataset.csv")
texts = [clean_text(t) for t in df["review"].tolist()]
labels = [1 if l == "positive" else 0 for l in df["sentiment"]]

# --- Train/val split ---
X_train_texts, X_val_texts, y_train, y_val = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# --- Text vectorization ---
max_words = 10000
max_len = 200
vectorizer = TextVectorization(
    max_tokens=max_words,
    output_sequence_length=max_len,
    standardize=None  # already cleaned
)
vectorizer.adapt(X_train_texts)

# Convert to vectorized tensors
X_train = vectorizer(tf.constant(X_train_texts))
X_val = vectorizer(tf.constant(X_val_texts))

# --- Define model ---
model = Sequential([
    Embedding(input_dim=max_words, output_dim=16, input_length=max_len),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# --- Train ---
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val))

# --- Save ---
model.save("baseline_model.keras")  # Keras 3 recommended format
print("Baseline model trained and saved.")
