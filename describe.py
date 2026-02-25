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
    percent25 = (ft_count(col)) * 0.25
    return ((1 - percent25 % 1) * sorted_col[percent25 // 1 - 1]) + ((percent25 % 1) * sorted_col[percent25 // 1])

def ft_quartile50(col):
    sorted_col = col.sort_values().reset_index(drop=True)
    percent50 = (ft_count(col)) * 0.5
    return (( 1 - percent50 % 1) * sorted_col[percent50 // 1 - 1]) + ((percent50 % 1) * sorted_col[percent50 // 1 ])

def ft_quartile75(col):
    sorted_col = col.sort_values().reset_index(drop=True)
    percent75 = (ft_count(col)) * 0.75
    return (( 1 - percent75 % 1) * sorted_col[percent75 // 1 - 1]) + ((percent75 % 1) * sorted_col[percent75 // 1])

def ft_max(col):
    max = col[0]
    for i in col:
        if (max < i):
            max = i
    return  max

def ft_range(col):
    return (ft_max(col) - ft_min(col))

def ft_iqr(col):
    return (ft_quartile75(col) - ft_quartile50(col))

def main():
	try:
		if (len(sys.argv) != 2):
			sys.exit("Wrong number of arguments")
		df = pd.read_csv(sys.argv[1], index_col=0)
	except (FileNotFoundError, PermissionError, pd.errors.EmptyDataError, pd.errors.ParserError):
		sys.exit("Incorrect path or file.")

	df = df.select_dtypes(include=np.number).dropna(axis=1, how='all')
	index = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max", "Range", "IQR"]
	try:
		df = df.agg([ft_count, ft_mean, ft_std, ft_min, ft_quartile25, ft_quartile50, ft_quartile75, ft_max, ft_range, ft_iqr])
	except Exception as e:
		sys.exit(f"Error reading dataset: {e}")
	df.index = index
	print(df.to_string())
	return

if __name__ == "__main__":
	main()
