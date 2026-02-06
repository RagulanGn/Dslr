import numpy as np
import pandas as pd
import sys

def ft_count(col) -> int:
    return (int) (col.notna().sum())

def ft_mean(col):
    return col.sum() / ft_count(col)

def ft_std(col):
    mean = ft_mean(col)
    d2 = abs(col - mean) ** 2
    return (d2.sum() / (ft_count(col) - 1)) ** 0.5

def ft_min(col):
    min = col[0]
    for i in col:
        if (min > i):
            min = i
    return  min

def ft_quartile25(col):
    sorted_col = col.sort_values().reset_index(drop=True)
    percent25 = (ft_count(col) - 1) * 0.25
    return ((1 - percent25 % 1) * sorted_col[percent25 // 1]) + ((percent25 % 1) * sorted_col[percent25 // 1 + 1])

def ft_quartile50(col):
    sorted_col = col.sort_values().reset_index(drop=True)
    percent50 = (ft_count(col) - 1) * 0.5
    return (( 1 - percent50 % 1) * sorted_col[percent50 // 1]) + ((percent50 % 1) * sorted_col[percent50 // 1 + 1])

def ft_quartile75(col):
    sorted_col = col.sort_values().reset_index(drop=True)
    percent75 = (ft_count(col) - 1) * 0.75
    return (( 1 - percent75 % 1) * sorted_col[percent75 // 1]) + ((percent75 % 1) * sorted_col[percent75 // 1 + 1])

def ft_max(col):
    max = col[0]
    for i in col:
        if (max < i):
            max = i
    return  max

def main(path:str):
	df = pd.read_csv(path, index_col=0)
	df2 = df.select_dtypes(include=np.number).dropna(axis=1, how='all')
	# print(df.to_string())
	print(df2.agg([ft_count, ft_mean, ft_std, ft_min, ft_quartile25, ft_quartile50, ft_quartile75, ft_max]).to_string())
	
	return

if __name__ == "__main__":
	# Check args
	main(sys.argv[1])