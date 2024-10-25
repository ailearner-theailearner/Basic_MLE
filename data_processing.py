import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

def load_and_split_data():
    try:
        # Load the data
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target

        # Split the data
        train_df, inference_df = train_test_split(df, test_size=0.2, random_state=42)

        # Save the data 
        train_df.to_csv('data/training.csv', index=False)
        inference_df.to_csv('data/inference.csv', index=False)
    except Exception as e:
        print(f"Error during data processing: {e}")

if __name__ == "__main__":
    load_and_split_data()