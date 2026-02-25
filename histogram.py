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
    # Parsing and Error Handling
    if len(sys.argv) != 2:
        print("Usage: python histogram.py <dataset_path>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    
    try:
        # Load dataset using Index as the index column
        df = pd.read_csv(dataset_path, index_col="Index")
    except Exception as e:
        print(f"Error reading dataset: {e}")
        sys.exit(1)

    # Filter numeric columns and exclude target/metadata
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Setup the plot grid (multi-plot layout)
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

    # Loop through all numerical features to create subplots
    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        
        # Plot distribution for each house
        for house in houses:
            # Clean data: Select house and remove NaNs
            data = df[df["Hogwarts House"] == house][col].dropna()
            
            # Use Seaborn for professional visualization
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
    
    # Hide empty subplots if the number of features is not a multiple of n_cols
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Final layout adjustments
    # plt.tight_layout()
    
    # Output management
    output_file = "histogram.png"
    plt.savefig(output_file)
    plt.show()
    print(f"Histograms saved successfully to {output_file}")
    
if __name__ == "__main__":
    main()