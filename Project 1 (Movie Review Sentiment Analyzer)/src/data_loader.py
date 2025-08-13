import pandas as pd

def load_imdb_dataset(file_path):
    """
    Load the IMDB dataset from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        texts (list of str): Movie reviews.
        labels (list of int): 1 for positive, 0 for negative.
    """
    df = pd.read_csv(file_path)
    
    # Check the first few rows to see column names
    print(df.head())

    # Assuming columns are 'review' and 'sentiment'
    texts = df['review'].tolist()
    labels = df['sentiment'].apply(lambda x: 1 if x.lower() == 'positive' else 0).tolist()
    
    return texts, labels

if __name__ == "__main__":
    texts, labels = load_imdb_dataset("data/imdb_dataset.csv")
    print(f"Loaded {len(texts)} reviews")
    print(f"First review: {texts[0]}")
    print(f"First label: {labels[0]}")