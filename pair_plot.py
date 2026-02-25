
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python pair_plot.py <dataset_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    
    try:
        df = pd.read_csv(dataset_path, index_col="Index")
    except Exception as e:
        print(f"Error reading dataset: {e}")
        sys.exit(1)

    df_clean = df.dropna()

    numeric_cols = df_clean.select_dtypes(include=['number']).columns.tolist()
    if not numeric_cols:
        print("No numeric columns found.")
        sys.exit(0)
    
    palette = {
        "Gryffindor": "#ae0001",
        "Slytherin": "#2a623d",
        "Ravenclaw": "#222f5b",
        "Hufflepuff": "#ffdb00"
    }
    print("Generating pair plot... (this might take a while)")
    
    # Pair plot
    sns.pairplot(
        df_clean[numeric_cols + ["Hogwarts House"]],
        hue="Hogwarts House",
        height=1.5,
        palette=palette
    )
    
    plt.savefig("pair_plot.png")
    plt.show()
    plt.close()
    print("Pair plot saved to pair_plot.png")

if __name__ == "__main__":
    main()