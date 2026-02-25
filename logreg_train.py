
import pandas as pd
import numpy as np
import sys
import os

# Features to drop based on visualization analysis
# - Arithmancy: Uniform distribution (homogeneous)
# - Care of Magical Creatures: Uniform distribution (homogeneous)
# - Defense Against the Dark Arts: Highly correlated with Astronomy (-1.0)
DROP_FEATURES = ["Arithmancy", "Care of Magical Creatures", "Defense Against the Dark Arts"]




def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, theta):
    return sigmoid(np.dot(X, theta))

def cost_function(X, y, theta):
    m = len(y)
    h = predict(X, theta)
    # Add epsilon to log to prevent log(0)
    epsilon = 1e-15
    cost = (-1 / m) * (np.dot(y.T, np.log(h + epsilon)) + np.dot((1 - y).T, np.log(1 - h + epsilon)))
    return cost

def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = []
    
    for i in range(iterations):
        h = predict(X, theta)
        gradient = (1 / m) * np.dot(X.T, (h - y))
        theta -= learning_rate * gradient
        #print(f"Iteration {i}, max gradient {np.max(np.abs(gradient))}")
        if i % 100 == 0:
            cost = cost_function(X, y, theta)
            cost_history.append(cost)
            # print(f"Iteration {i}: Cost {cost}")
            
    return theta, cost_history

def preprocess_data(df, train=True, stats=None):
    # Filter numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Remove index if present in numeric_cols (usually index is not in select_dtypes if it's index, but check)
    if "Index" in numeric_cols:
        numeric_cols.remove("Index")
        
    # Drop unwanted features
    selected_features = [f for f in numeric_cols if f not in DROP_FEATURES]
    X_df = df[selected_features]
    print("Nombre de features :", len(selected_features))
    print("Features :", selected_features)
    
    # Handle missing values: Fill with mean
    # If train, calculate mean and save it. If test, use saved mean.
    if train:
        means = X_df.mean()
        # Save means for later? we'll return them
        X_df = X_df.fillna(means)
        
        # Normalize (Standardization): (x - mean) / std
        stds = X_df.std()
        X_normalized = (X_df - means) / stds
        
        stats = {"means": means, "stds": stds, "features": selected_features}
    else:
        if stats is None:
            raise ValueError("Stats must be provided for test data")
        # Ensure we use the exact same features
        X_df = X_df[stats["features"]] 
        X_df = X_df.fillna(stats["means"])
        X_normalized = (X_df - stats["means"]) / stats["stds"]
        
    X_numpy = X_normalized.values
    
    # Add intercept term (column of ones)
    m = X_numpy.shape[0]
    X_with_intercept = np.hstack((np.ones((m, 1)), X_numpy))
    
    return X_with_intercept, stats

def main():
    if len(sys.argv) != 2:
        print("Usage: python logreg_train.py <dataset_path>")
        sys.exit(1)
    
    dataset_path = sys.argv[1]
    
    try:
        df = pd.read_csv(dataset_path, index_col="Index")
    except Exception as e:
        print(f"Error reading dataset: {e}")
        sys.exit(1)

    
    print("Preprocessing data...")
    X, stats = preprocess_data(df, train=True)
    
    # Target variable
    y_raw = df["Hogwarts House"]
    houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    
    # Train One-vs-All
    all_weights = {} # {house: theta}
    
    learning_rate = 0.1 # Tweak as needed
    iterations = 10000   # Tweak as needed
    
    print(f"Training models with LR={learning_rate}, Iterations={iterations}...")
    
    for house in houses:
        print(f"Training for {house}...")
        # Create binary target: 1 if house, 0 otherwise
        y_binary = (y_raw == house).astype(int).values
        
        # Initialize theta
        n_features = X.shape[1]
        theta = np.zeros(n_features)
        
        theta_trained, _ = gradient_descent(X, y_binary, theta, learning_rate, iterations)
        all_weights[house] = theta_trained

    np.savez("weights.npz", 
             weights=all_weights, 
             means=stats["means"].values, 
             stds=stats["stds"].values, 
             features=stats["features"])
    
    print("Training complete. Weights saved to weights.npz")

if __name__ == "__main__":
    main()
