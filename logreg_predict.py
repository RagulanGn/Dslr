import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def main(dataset_path, weight_path):
	weight = pd.read_csv(weight_path, index_col=0)
	mean = weight["mean"]
	weight.drop("mean", axis=1, inplace=True)
	std = weight["std"]
	weight.drop("std", axis=1, inplace=True)
	
	dataset = pd.read_csv(dataset_path, index_col=0)
	dataset = dataset.dropna(axis=1, how='all')
	dataset = dataset[int(len(dataset.index) * 0.8):]
	hogwart_house = 'Hogwarts House' in dataset
	if (hogwart_house):
		dataset.set_index("Hogwarts House", append=True, inplace=True) #Add Hogwart House as index
	#Should I drop NaN rows
	dataset = dataset.select_dtypes(include=np.number) #Drop every NaN columns and any NaN row
	# dataset = dataset.drop(columns=["Arithmancy", "Astronomy", "Care of Magical Creatures", "Potions"])
	dataset = dataset.drop(columns=["Arithmancy", "Muggle Studies", "History of Magic","Care of Magical Creatures", "Potions"])

	dataset.fillna(mean, inplace=True)
	dataset = (dataset - mean) / std #Standardization to avoid overflow
	dataset["biais"] = 1
	# sigmoide (Dataset . weight) (produit scalaire) => Plus x est grand ou petit plus le resultat est sur
	# Donc faire l'operation sur les 4 classe (4 poids) et prendre le plus grand
	# Pas besoin de la sigmoide car on travaille sur une fonction croissante
	
 
	result = dataset @ weight
	result = result.rename(columns={1: "Hogwarts House"})
	result = result.idxmax(axis=1)
	result.to_csv("result.csv")
	print(result.to_string())
	if (hogwart_house):
		index = dataset.index.get_level_values("Hogwarts House").to_numpy()
		print(f"Score based on training set : {accuracy_score(result.values, index)}")
	return

if __name__ == "__main__":
	main(sys.argv[1], sys.argv[2])

# Fill Both with 0 (0.961875)
# Fill only test 0 (0.960625)
# Fill both with mean (0.9625)
# Fill train with mean and test with 0 (0.98)


# avec potion + 3 autres (0.98) Full set
# dataset = dataset.drop(columns=["Arithmancy", "Astronomy", "Care of Magical Creatures", "Potions"])

#0.9718