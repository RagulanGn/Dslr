import sys
import pandas as pd
import numpy as np
import traceback

# Descente de gradient => Fonction a minimiser => Sigmoide : 1 /(1 - exp(-lambda * x)) Avec x = a * P1 + b * P2 + c * P3 + ... , PX les parametres
# Regression logistique 1 vs all => 4 sigmoide independante

# Courbe sigmoide => Double asymptote horinzontale a y=1 et y=-1

#If we want to start the b at 0 we need to take care of dividing by 0
def sigmoide(x):
    x_safe = np.clip(x, -500, 500)
    return (1/(1 - np.exp(x_safe)))

def main(path):
	df1 = pd.read_csv(path, index_col=0)
	df2 = df1.set_index("Hogwarts House", append=True) #Add Hogwart House as index
	# df = df2.select_dtypes(include=np.number).dropna(axis=1, how='all') #Drop every NaN columns
	df = df2.select_dtypes(include=np.number).dropna(axis=1, how='all') #Drop every NaN columns
	df = df.fillna(df.mean())
	#Solution against NaN => drop the entire row. Should I found a better solution ?
	mean = df.mean()
	std = df.std()
	df = (df - mean) / std #Standardization to avoid overflow
	# print(df)
	a = 0.01
	b1 = np.ones(13) * 0.1
	b2 = np.ones(13) * 0.1
	b3 = np.ones(13) * 0.1
	b4 = np.ones(13) * 0.1
	i = 0
	n = 13
	index = df.index.get_level_values("Hogwarts House").to_numpy()
	while (True) :
		b1 = b1 - (a/n) * (df.T @ (sigmoide(df @ b1) - (index == "Ravenclaw").astype(int)))
		b2 = b2 - (a/n) * (df.T @ (sigmoide(df @ b2) - (index == "Slytherin").astype(int)))
		b3 = b3 - (a/n) * (df.T @ (sigmoide(df @ b3) - (index == "Gryffindor").astype(int)))
		b4 = b4 - (a/n) * (df.T @ (sigmoide(df @ b4) - (index == "Hufflepuff").astype(int)))
		if i == 1000 :
			break
		i += 1
	weight = pd.DataFrame({"Ravenclaw": b1, "Slytherin": b2, "Gryffindor":b3, "Hufflepuff":b4, "mean":mean, "std":std})
	# weight = pd.DataFrame({"Ravenclaw": b1, "Slytherin": b2, "Gryffindor":b3, "Hufflepuff":b4})
	weight.to_csv("weight.csv")
	return

if __name__ == "__main__":
	try:
		main(sys.argv[1])
	except Exception:
		traceback.print_exc()
