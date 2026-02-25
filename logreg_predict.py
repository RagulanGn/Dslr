
import pandas as pd
import numpy as np
import sys
import os

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_proba(X, theta):
    return sigmoid(np.dot(X, theta))

def load_model(weights_path):
    # Load weights and stats
    data = np.load(weights_path, allow_pickle=True)
    weights = data['weights'].item()
    means = data['means']
    stds = data['stds']
    features = data['features']

    return weights, means, stds, features

def preprocess_test_data(df, means, stds, features):
    
    # Select features
    X_df = df[features]
    
    means_series = pd.Series(means, index=features)
    stds_series = pd.Series(stds, index=features)
    
    X_df = X_df.fillna(means_series)
    
    # Normalize
    X_normalized = (X_df - means_series) / stds_series
    X_numpy = X_normalized.values
    
    # Add intercept
    m = X_numpy.shape[0]
    X_with_intercept = np.hstack((np.ones((m, 1)), X_numpy))
    
    return X_with_intercept

def main():
    if len(sys.argv) != 3:
        print("Usage: python logreg_predict.py <dataset_path> <weights_path>")
        sys.exit(1)
        
    dataset_path = sys.argv[1]
    weights_path = sys.argv[2]
    
    try:
        df = pd.read_csv(dataset_path, index_col="Index")
    except Exception as e:
        print(f"Error reading dataset: {e}")
        sys.exit(1)

    try:
        weights, means, stds, features = load_model(weights_path)
    except Exception as e:
        print(f"Error loading weights: {e}")
        sys.exit(1)
        
    print("Preprocessing test data...")
    try:
        X = preprocess_test_data(df, means, stds, features)
    except Exception as e:
         print(f"Error during preprocessing: {e}")
         sys.exit(1)
    

    # Predict for each house
    print("Making predictions...")
    houses = list(weights.keys())

    print("houses ",houses)
    # Try providing a defined order if possible, or just keys.
    # The output needs house name.
    np.set_printoptions(suppress=True)
    # Calculate probability for each house
    probs = {}
    for house, theta in weights.items():
        probs[house] = predict_proba(X, theta)

    final_predictions = []
    # Iterate over samples
    m = X.shape[0]
    for i in range(m):
        best_house = None
        best_prob = -1.0
        for house in houses:
            p = probs[house][i]
            if p > best_prob:
                best_prob = p
                best_house = house
        final_predictions.append(best_house)
        
    # Create output dataframe
    output_df = pd.DataFrame({
        "Index": df.index,
        "Hogwarts House": final_predictions
    })
    
    # Save to houses.csv
    output_df.to_csv("houses.csv", index=False)
    print("Predictions saved to houses.csv")

if __name__ == "__main__":
    main()
