import pandas as pd
from typing import Tuple

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert 'review' in df.columns and 'sentiment' in df.columns, "CSV must contain 'review' and 'sentiment' columns"
    return df

def train_test_split(df: pd.DataFrame, test_size: float=0.2, random_state: int=42) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    from sklearn.model_selection import train_test_split as tts
    X = df['review']
    y = df['sentiment']
    return tts(X, y, test_size=test_size, random_state=random_state)
