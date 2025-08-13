# preprocessing.py
# ---------------------------------------------------------------
# Text cleaning and preprocessing utilities for sentiment models.
# Works with Keras 3 (TensorFlow backend).
#
# - Clean text (strip HTML, non-letters, lowercase)
# - Split train/validation with stratification
# - Create and adapt a TextVectorization layer efficiently
# - Return either vectorized tensors or tf.data datasets
# - Extras: export vocab, save/load the vectorizer

from __future__ import annotations

import re
from typing import Iterable, List, Tuple, Union, Optional

import tensorflow as tf
import keras
from keras.layers import TextVectorization
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    """Remove HTML tags, non-letter characters, and lowercase the text."""
    text = re.sub(r"<.*?>", "", text)         # remove HTML tags
    text = re.sub(r"[^a-zA-Z ]", "", text)    # keep only letters and spaces
    return text.lower()


def encode_labels(labels: Iterable[str]) -> List[int]:
    """Map 'positive'->1 and everything else->0."""
    return [1 if str(label).strip().lower() == "positive" else 0 for label in labels]


def build_vectorizer(
    max_words: int = 10_000,
    max_len: int = 200,
    standardize: Optional[str] = None,
) -> TextVectorization:
    """
    Create a TextVectorization layer. We keep standardize=None because we already clean.
    """
    return TextVectorization(
        max_tokens=max_words,
        output_sequence_length=max_len,
        standardize=standardize,  # None -> we already cleaned text
    )


def adapt_vectorizer(vectorizer: TextVectorization, texts: Iterable[str], batch_size: int = 512) -> None:
    """
    Efficiently adapt the vectorizer using a batched tf.data pipeline.
    """
    ds = tf.data.Dataset.from_tensor_slices(list(texts)).batch(batch_size)
    vectorizer.adapt(ds)


def make_datasets(
    vectorizer: TextVectorization,
    X_train_texts: List[str],
    y_train: List[int],
    X_val_texts: List[str],
    y_val: List[int],
    batch_size: int = 32,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Build tf.data pipelines that vectorize on the fly.
    """
    autotune = tf.data.AUTOTUNE

    def to_ds(x_texts: List[str], y: List[int], training: bool) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((x_texts, y))
        if training:
            ds = ds.shuffle(buffer_size=min(len(x_texts), 10_000), seed=42)
        ds = ds.map(
            lambda x, yy: (vectorizer(x), tf.cast(yy, tf.int32)),
            num_parallel_calls=autotune,
        )
        return ds.batch(batch_size).prefetch(autotune)

    return to_ds(X_train_texts, y_train, True), to_ds(X_val_texts, y_val, False)


def preprocess_data(
    texts: Iterable[str],
    labels: Iterable[str],
    max_words: int = 10_000,
    max_len: int = 200,
    test_size: float = 0.2,
    batch_size: int = 32,
    return_datasets: bool = False,
) -> Union[
    Tuple[tf.Tensor, tf.Tensor, List[int], List[int], TextVectorization],
    Tuple[tf.data.Dataset, tf.data.Dataset, TextVectorization]
]:
    """
    Clean, vectorize, encode labels, and split data into train/val.

    Args:
        texts: Raw text iterable.
        labels: String labels ('positive'/'negative' etc.).
        max_words: Vocabulary size for the vectorizer.
        max_len: Fixed sequence length.
        test_size: Validation split fraction.
        batch_size: Batch size for dataset mode.
        return_datasets: If True, return (train_ds, val_ds, vectorizer).
                         If False (default), return vectorized tensors:
                         (X_train, X_val, y_train, y_val, vectorizer).

    Returns:
        If return_datasets is False:
            X_train (tf.Tensor), X_val (tf.Tensor), y_train (List[int]), y_val (List[int]), vectorizer
        If return_datasets is True:
            train_ds (tf.data.Dataset), val_ds (tf.data.Dataset), vectorizer
    """
    # Clean texts and encode labels
    texts = [clean_text(str(t)) for t in texts]
    y = encode_labels(labels)

    # Train/val split with stratification for balanced classes
    X_train_texts, X_val_texts, y_train, y_val = train_test_split(
        texts, y, test_size=test_size, random_state=42, stratify=y
    )

    # Vectorizer
    vectorizer = build_vectorizer(max_words=max_words, max_len=max_len, standardize=None)
    adapt_vectorizer(vectorizer, X_train_texts)

    if return_datasets:
        train_ds, val_ds = make_datasets(vectorizer, X_train_texts, y_train, X_val_texts, y_val, batch_size)
        return train_ds, val_ds, vectorizer
    else:
        # Vectorize eagerly into tensors (drop-in behavior with your original code)
        X_train = vectorizer(tf.constant(X_train_texts))
        X_val = vectorizer(tf.constant(X_val_texts))
        return X_train, X_val, y_train, y_val, vectorizer


# ----------------------- Optional utilities -----------------------

def export_vocabulary(vectorizer: TextVectorization) -> List[str]:
    """Return the learned vocabulary as a Python list."""
    return list(vectorizer.get_vocabulary())


def save_vectorizer(vectorizer: TextVectorization, path: str) -> None:
    """
    Save the TextVectorization layer as a Keras object.
    Note: This saves config + weights (including vocab) with Keras 3 format.
    """
    # Wrap in a trivial model for robust saving/loading
    inp = keras.Input(shape=(1,), dtype=tf.string)
    out = vectorizer(inp)
    wrapper = keras.Model(inp, out)
    wrapper.save(path)


def load_vectorizer(path: str) -> TextVectorization:
    """
    Load a previously saved TextVectorization wrapper model and return the layer.
    """
    wrapper = keras.models.load_model(path)
    # The first (and only) layer after Input should be TextVectorization
    for layer in wrapper.layers:
        if isinstance(layer, TextVectorization):
            return layer
    raise ValueError("No TextVectorization layer found in loaded model.")
