import sys
import numpy as np
import pandas as pd


def main(dataset_path, weight_path):
	weight = pd.read_csv(weight_path, index_col=0)
	dataset = pd.read_csv(dataset_path, index_col=0)
	dataset.set_index("Hogwarts House", append=True, inplace=True) #Add Hogwart House as index
	#Should not drop NaN row, I have to handle it
	dataset = dataset.select_dtypes(include=np.number).dropna(axis=1, how='all').dropna() #Drop every NaN columns and any NaN row
	dataset = (dataset - dataset.mean()) / dataset.std() #Standardization to avoid overflow
	
	# sigmoide (Dataset . weight) (produit scalaire) => Plus x est grand ou petit plus le resultat est sur
	# Donc faire l'operation sur les 4 classe (4 poids) et prendre le plus grand
	# Pas besoin de la sigmoide car on travaille sur une fonction croissante
	
	result = dataset @ weight
	result.to_csv("result.csv")
	result = result.idxmax(axis=1)
	print(result.to_string())
	index = dataset.index.get_level_values("Hogwarts House").to_numpy()
	score = (result.values == index)
	print(f"Score based on training set : {sum(score)/ len(result.values)}")
	return

if __name__ == "__main__":
	main(sys.argv[1], sys.argv[2])
