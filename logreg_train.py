import sys
import pandas as pd
import numpy as np
import traceback

# Descente de gradient => Fonction a minimiser => Sigmoide : 1 /(1 - exp(-lambda * x)) Avec x = a * P1 + b * P2 + c * P3 + ... , PX les parametres
# Regression logistique 1 vs all => 4 sigmoide independante

# Courbe sigmoide => Double asymptote horinzontale a y=1 et y=-1

def ft_count(col):
    return (col.notna().sum())

def ft_mean(col):
    return col.sum() / ft_count(col)

def ft_std(col):
    mean = ft_mean(col)
    d2 = abs(col - mean) ** 2
    return (d2.sum() / (ft_count(col) - 1)) ** 0.5

def sigmoide(x):
    """ Return the value of the sigmoide function at point x """
    x_safe = np.clip(x, -500, 500)
    return (1/(1 + np.exp(-x_safe)))

def main():
	try:
		if len(sys.argv) != 2:
			print("No dataset path, usage : python program_name dataset_path")
			sys.exit(1)
		path = sys.argv[1]
		df1 = pd.read_csv(path, index_col=0)

	except (FileNotFoundError, PermissionError, pd.errors.EmptyDataError, pd.errors.ParserError):
		print("Incorrect path or file.")
		sys.exit(1)

	# df1 = df1.sample(frac=1, random_state=46).reset_index(drop=True) # Reset index to avoid issues with loc/iloc mixing

	df2 = df1.set_index("Hogwarts House", append=True)																		#Append Hogwart House as index (Multi-Index)
	df = df2.select_dtypes(include=np.number).dropna(axis=1, how='all')														#Drop every NaN columns 
	df = df.drop(columns=["Arithmancy", "Care of Magical Creatures"])														#Drop 2 columns because they are not vector of any information but create noise
	df = df.fillna(df.median())																								#Replace NaN with median
 
	# mean = df.mean()
	# std = df.std()
	mean = ft_mean(df)
	std = ft_std(df)
	df = (df - mean) / std																									#Standardization to avoid overflow
	# df = df[:int(len(df.index) * 0.8)]																					#Take 80% of train to allows to evaluate our model with the other 20% (Calculate accuracy score)
	df.insert(0, 'bias', 1)																									#Bias to not force our model to pass by origin

	a = 0.001																												#Learning rate
	b1 = b2 = b3 = b4 = np.zeros(len(df.columns))																			#Initialize all b vector of size = len(features) with 0
	i = 0
	n = len(df.index)																										#Number of line in our dataset => Allow us to calculate the mean of the error in each step
	index = df.index.get_level_values("Hogwarts House").to_numpy()															#Numpy array of indexes (Hogwart House of each student)
	while (i < 1000) :
		b1 = b1 - (a/n) * (df.T @ (sigmoide(df @ b1) - (index == "Ravenclaw").astype(int)))									#Correction of parameters corresponding to Ravenclaw
		b2 = b2 - (a/n) * (df.T @ (sigmoide(df @ b2) - (index == "Slytherin").astype(int)))
		b3 = b3 - (a/n) * (df.T @ (sigmoide(df @ b3) - (index == "Gryffindor").astype(int)))
		b4 = b4 - (a/n) * (df.T @ (sigmoide(df @ b4) - (index == "Hufflepuff").astype(int)))
		i += 1
	weight = pd.DataFrame({"Ravenclaw": b1, "Slytherin": b2, "Gryffindor": b3, "Hufflepuff": b4, "mean": mean, "std": std}) # Create a dataframe with the 4 weight combined + mean and standard deviation
	weight.to_csv("weight.csv") #Save to csv
	return

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.exit(f"Error : {e}")