import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

def calculate_mean(data):
    """Calculates the arithmetic mean of a numeric sequence.

    Args:
        data (np.array): A sequence of numerical values.

    Returns:
        float: The mean value.
    """
    if len(data) == 0:
        return 0.0
    return sum(data) / len(data)

def calculate_correlation(x, y):
    """Calculates the Pearson correlation coefficient between two variables.

    This function excludes NaN values and computes the correlation manually
    to adhere to project constraints regarding heavy-lifting functions.

    Args:
        x (np.array): First numerical sequence.
        y (np.array): Second numerical sequence.

    Returns:
        float: Pearson correlation coefficient ranging from -1 to 1.
    """
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_c = x[mask]
    y_c = y[mask]
    
    if len(x_c) < 2:
        return 0.0

    mu_x = calculate_mean(x_c)
    mu_y = calculate_mean(y_c)

    numerator = sum((x_c - mu_x) * (y_c - mu_y))
    denominator = np.sqrt(sum((x_c - mu_x)**2) * sum((y_c - mu_y)**2))

    if denominator == 0:
        return 0.0
    return numerator / denominator

def main():
    """Main execution block: parses data, finds similar features, and plots."""
    if len(sys.argv) != 2:
        print("Usage: python scatter_plot.py <dataset_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    
    try:
        df = pd.read_csv(dataset_path, index_col="Index")
    except Exception as e:
        print(f"Error reading dataset: {e}")
        sys.exit(1)

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    max_corr = -1.0
    best_pair = (None, None)

    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            f1, f2 = numeric_cols[i], numeric_cols[j]
            corr_val = calculate_correlation(df[f1].values, df[f2].values)
            
            if abs(corr_val) > max_corr:
                max_corr = abs(corr_val)
                best_pair = (f1, f2)

    feat1, feat2 = best_pair
    print(f"Features identified: '{feat1}' and '{feat2}'")
    print(f"Pearson Correlation Coefficient: {max_corr:.4f}")

    palette = {
        "Gryffindor": "#ae0001",
        "Slytherin": "#2a623d",
        "Ravenclaw": "#222f5b",
        "Hufflepuff": "#ffdb00"
    }

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=df, 
        x=feat1, 
        y=feat2, 
        hue="Hogwarts House", 
        palette=palette, 
        alpha=0.6,
        edgecolor='w'
    )

    plt.title(f"Scatter Plot: {feat1} vs {feat2}\n(Highest correlation detected)")
    plt.xlabel(feat1)
    plt.ylabel(feat2)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    output_name = "scatter_plot.png"
    plt.savefig(output_name)
    plt.show()
    print(f"Plot saved as {output_name}")
    

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.exit(f"Error : {e}")