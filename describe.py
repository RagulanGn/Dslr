import numpy as np
import pandas as pd
import sys

def ft_count(col) -> int:
    return (int) (col.notna().sum())

def ft_mean(col):
    return col.sum() / ft_count(col)

def ft_std(col):
    return

def ft_min(col):
    return 

def ft_quartile25(col):
    return

def ft_quartile50(col):
    return

def ft_quartile75(col):
    return

def ft_max(col):
    return 

def main(path:str):
	df = pd.read_csv(path, index_col=0)
	df2 = df.select_dtypes(include=np.number).dropna(axis=1, how='all')
	# print(df.to_string())
	print(df2.agg([ft_count, ft_mean, "mean"]).to_string())
	
	return

if __name__ == "__main__":
	# Check args
	main(sys.argv[1])