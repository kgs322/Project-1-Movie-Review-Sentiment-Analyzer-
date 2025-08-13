from transformers import BertTokenizer, TFBertForSequenceClassification, create_optimizer
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

# Optional: from preprocessing import clean_text

# Load dataset
df = pd.read_csv("IMDB Dataset.csv")
texts = df["review"].tolist()
labels = [1 if x == 'positive' else 0 for x in df["sentiment"]]

# Train/validation split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# Prepare TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
)).shuffle(1000).batch(16)

val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    val_labels
)).batch(16)

# Load model
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Optimizer & schedule
num_train_steps = len(train_dataset) * 3
optimizer, schedule = create_optimizer(init_lr=5e-5, num_warmup_steps=100, num_train_steps=num_train_steps)

# Compile model
model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])

# Train model
model.fit(train_dataset, validation_data=val_dataset, epochs=3)

# Save model and tokenizer
model.save_pretrained("transformer_model")
tokenizer.save_pretrained("transformer_model")
print("Transformer model trained and saved.")
