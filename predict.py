import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

def main():
	try:
		if len(sys.argv) != 3:
			print("No dataset path, usage : python program_name dataset_path weight_path")
			sys.exit(1)
		dataset_path = sys.argv[1]
		weight_path = sys.argv[2]
		dataset = pd.read_csv(dataset_path, index_col=0)
		weight = pd.read_csv(weight_path, index_col=0)

	except (FileNotFoundError, PermissionError, pd.errors.EmptyDataError, pd.errors.ParserError):
		print("Incorrect path or file.")
		sys.exit(1)
	
	mean = weight["mean"]
	std = weight["std"]
	weight.drop("mean", axis=1, inplace=True)
	weight.drop("std", axis=1, inplace=True)
	

	dataset = dataset.dropna(axis=1, how='all')
 
	hogwart_house = 'Hogwarts House' in dataset											#If training set
	if (hogwart_house):
		dataset.set_index("Hogwarts House", append=True, inplace=True)					#Add Hogwart House as index
	
	dataset = dataset.select_dtypes(include=np.number)
	dataset = dataset.drop(columns=["Arithmancy", "Care of Magical Creatures"], errors='ignore')

	dataset.fillna(mean, inplace=True)													#Replace na by mean of training dataset
	dataset = (dataset - mean) / std													#Standardization to avoid overflow
	dataset["bias"] = 1																	#Redefine bias (it became Nan because of standardization on dataframe)

	result = dataset @ weight															# Matrix Muliplication to apply weight to each features of student => result his a (?, 4) each column as a score the higher the score is more likely the student belong to class
	
	result = result.idxmax(axis=1)														# Give the index of the max value along rows in a Series
	result.name = "Hogwarts House"														# Rename Series to have the correct format
	if (hogwart_house):																	#If training set
		index = dataset.index.get_level_values("Hogwarts House").to_numpy()
		print(f"Score based on training set : {accuracy_score(result.values, index)}")
		result = result.reset_index(level=1, drop=True)
	result.to_csv("houses.csv")
	return

if __name__ == "__main__":
	try:
		main()
	except Exception as e:
		sys.exit(f"Error : {e}")
