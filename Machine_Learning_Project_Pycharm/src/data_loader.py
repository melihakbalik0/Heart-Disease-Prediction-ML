import pandas as pd

def load_heart_data(path):
    """
    Loads the dataset
    """
    df = pd.read_csv(path)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    return df