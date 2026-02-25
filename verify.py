
import pandas as pd
import numpy as np
import sys
from logreg_train import preprocess_data, gradient_descent, DROP_FEATURES
from logreg_predict import predict_proba, preprocess_test_data

def accuracy_score(y_true, y_pred):
    correct = 0
    total = len(y_true)
    for t, p in zip(y_true, y_pred):
        if t == p:
            correct += 1
    return correct / total

def main():
    if len(sys.argv) != 2:
        print("Usage: python verify.py <dataset_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    
    try:
        df = pd.read_csv(dataset_path, index_col="Index")
    except Exception as e:
        print(f"Error reading dataset: {e}")
        sys.exit(1)

    # Shuffle dataset
    df = df.sample(frac=1, random_state=49).reset_index(drop=True) # Reset index to avoid issues with loc/iloc mixing
    
    # Split 80/20
    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    
    print(f"Training on {len(train_df)} samples, Validating on {len(test_df)} samples...")
    
    # Train
    X_train, stats = preprocess_data(train_df, train=True)
    y_raw_train = train_df["Hogwarts House"]
    houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    
    all_weights = {}
    learning_rate = 0.1 # Lower learning rate for stability
    iterations = 1   # More iterations for convergence
    
    for house in houses:
        y_binary = (y_raw_train == house).astype(int).values
        n_features = X_train.shape[1]
        theta = np.zeros(n_features)
        theta_trained, _ = gradient_descent(X_train, y_binary, theta, learning_rate, iterations)
        all_weights[house] = theta_trained

    probs_train = {}
    for house, theta in all_weights.items():
        probs_train[house] = predict_proba(X_train, theta)
        
    train_predictions = []
    m_train = X_train.shape[0]
    for i in range(m_train):
        best_house = None
        best_prob = -1.0
        for house in houses:
            p = probs_train[house][i]
            if p > best_prob:
                best_prob = p
                best_house = house
        train_predictions.append(best_house)
    
    acc_train = accuracy_score(y_raw_train.values, train_predictions)
    print(f"Training Accuracy: {acc_train * 100:.2f}%")

    means = stats["means"]
    stds = stats["stds"]
    features = stats["features"]
    
    X_test = preprocess_test_data(test_df, means, stds, features)
    
    probs = {}
    for house, theta in all_weights.items():
        probs[house] = predict_proba(X_test, theta)
        
    final_predictions = []
    m = X_test.shape[0]
    for i in range(m):
        best_house = None
        best_prob = -1.0
        for house in houses:
            p = probs[house][i]
            if p > best_prob:
                best_prob = p
                best_house = house
        final_predictions.append(best_house)
        
    # Calculate accuracy
    y_true = test_df["Hogwarts House"].values
    acc = accuracy_score(y_true, final_predictions)
    
    print(f"Validation Accuracy: {acc * 100:.15f}%")
    
    if acc >= 0.98:
        print("SUCCESS: Accuracy is >= 98%")
    else:
        print("FAILURE: Accuracy is < 98%")

if __name__ == "__main__":
    main()
