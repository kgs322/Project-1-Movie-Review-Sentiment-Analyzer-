import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean_text(text):
    text = re.sub(r"<.*?>", "", text)  # remove HTML tags
    text = re.sub(r"[^a-zA-Z ]", "", text)  # keep only letters and spaces
    return text.lower()

def preprocess_data(texts, labels, max_words=10000, max_len=200, test_size=0.2):
    # Clean texts
    texts = [clean_text(t) for t in texts]
    
    # Tokenize
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len)
    
    # Encode labels
    labels = [1 if label == "positive" else 0 for label in labels]
    
    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(padded, labels, test_size=test_size, random_state=42)
    
    return X_train, X_val, y_train, y_val, tokenizer
