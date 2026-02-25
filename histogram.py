import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys

def main():
    """Main execution block: parses data, generates histograms for all courses.

    This script identifies which Hogwarts course has a homogeneous score 
    distribution between all four houses by visualizing all numerical features.
    It uses official house colors and professional plotting standards.
    """
    if len(sys.argv) != 2:
        print("Usage: python histogram.py <dataset_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    
    try:
        df = pd.read_csv(dataset_path, index_col="Index")
    except Exception as e:
        print(f"Error reading dataset: {e}")
        sys.exit(1)

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    n_cols = 4
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 5 * n_rows), gridspec_kw={"wspace": 0.3, "hspace": 0.4})
    axes = axes.flatten()
    
    houses = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]
    
    palette = {
        "Gryffindor": "#ae0001",
        "Slytherin": "#2a623d",
        "Ravenclaw": "#222f5b",
        "Hufflepuff": "#ffdb00"
    }

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        
        for house in houses:
            data = df[df["Hogwarts House"] == house][col].dropna()
            
            sns.histplot(
                data, 
                ax=ax, 
                label=house, 
                color=palette.get(house, 'gray'), 
                kde=True, 
                element="step", 
                alpha=0.3
            )
        
        ax.set_title(f"Distribution of {col}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Scores")
        ax.set_ylabel("Frequency")
        ax.legend(prop={'size': 8})
    
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    
    output_file = "histogram.png"
    plt.savefig(output_file)
    plt.show()
    print(f"Histograms saved successfully to {output_file}")
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.exit(f"Error : {e}")