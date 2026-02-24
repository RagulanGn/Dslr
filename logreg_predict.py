import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

def print_error_class(result, dataset):
	""" Custom function to display all the difference between prediction and reality"""
	index = dataset.index.get_level_values("Hogwarts House").to_numpy()
	for i in range(len(index)):
		if ((result.values[i] != index[i])):
			print(f"dataset : {index[i]} / prediction : {result.values[i]}")

# sigmoide (Dataset . weight) (produit scalaire) => Plus x est grand ou petit plus le resultat est sur
# Donc faire l'operation sur les 4 classe (4 poids) et prendre le plus grand
# Pas besoin de la sigmoide car on travaille sur une fonction croissante

def main():
	try:
		if len(sys.argv) != 2:
			print("No dataset path, usage : python program_name dataset_path")
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
	# dataset = dataset[int(len(dataset.index) * 0.8):]
 
	hogwart_house = 'Hogwarts House' in dataset											#If training set
	if (hogwart_house):
		dataset.set_index("Hogwarts House", append=True, inplace=True)					#Add Hogwart House as index
	
	dataset = dataset.select_dtypes(include=np.number)
	dataset = dataset.drop(columns=["Arithmancy", "Care of Magical Creatures"])

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
		# print_error_class(result, dataset)
	result.to_csv("houses.csv")
	return

if __name__ == "__main__":
	main()



#--------------------------------------------------------------------------------------------------------------------------------------------------------#

# Fill Both with 0 (0.961875)
# Fill only test 0 (0.960625)
# Fill both with mean (0.9625)
# Fill train with mean and test with 0 (0.98)


# avec potion + 3 autres (0.98) Full set
# dataset = dataset.drop(columns=["Arithmancy", "Astronomy", "Care of Magical Creatures", "Potions"])

#0.9718


# dataset : Slytherin / prediction : Ravenclaw
# dataset : Slytherin / prediction : Ravenclaw
# dataset : Slytherin / prediction : Ravenclaw
# dataset : Slytherin / prediction : Hufflepuff
# dataset : Gryffindor / prediction : Hufflepuff
# dataset : Gryffindor / prediction : Hufflepuff
# dataset : Gryffindor / prediction : Ravenclaw
# dataset : Gryffindor / prediction : Ravenclaw
# dataset : Ravenclaw / prediction : Hufflepuff

# 4 * Ravenclaw instead of 	Slytherin
# 1 * Hufflepuff instead of Slytherin
# 2 * Hufflepuff instead of Gryffindor
# 2 * Ravenclaw instead of  Gryffindor
# 1 * Hufflepuff instead of Ravenclaw

# ==> Meaning my model is detecting too much Ravenclaw and Hufflepuff ==> maybe because they are overrepresented inside th training