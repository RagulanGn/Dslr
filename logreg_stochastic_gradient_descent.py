import sys
import pandas as pd
import numpy as np
import traceback
import random

# Descente de gradient Stochastique -> Update des parametres b apres chaque data
# !!Attention il vaut mieux random l ordre du tableau que l index pour etre sur que toute les datas est une influence!!

def sigmoide(x):
    x_safe = np.clip(x, -500, 500)
    return (1/(1 - np.exp(x_safe)))

def main(path):
	df1 = pd.read_csv(path, index_col=0)
	df2 = df1.set_index("Hogwarts House", append=True) #Add Hogwart House as index
	# df = df2.select_dtypes(include=np.number).dropna(axis=1, how='all') #Drop every NaN columns
	df = df2.select_dtypes(include=np.number).dropna(axis=1, how='all').dropna() #Drop every NaN columns
	#Solution against NaN => drop the entire row. Should I found a better solution ?
	mean = df.mean()
	std = df.std()
	df = (df - mean) / std #Standardization to avoid overflow
	print(df)
	a = 0.1
	b1 = np.ones(13) * 0.1
	b2 = np.ones(13) * 0.1
	b3 = np.ones(13) * 0.1
	b4 = np.ones(13) * 0.1
	i = 0
	n = 13
	index = df.index.get_level_values("Hogwarts House").to_numpy()
	while (True) :
		random_index = random.randint(0,len(df.index) - 1)
		random_student = df.iloc[random_index]
		# print(index[random_index])
		b1 = b1 - a * (random_student * (sigmoide(random_student @ b1) - (int(index[random_index] == "Ravenclaw"))))
		b2 = b2 - a * (random_student * (sigmoide(random_student @ b2) - (int(index[random_index] == "Slytherin"))))
		b3 = b3 - a * (random_student * (sigmoide(random_student @ b3) - (int(index[random_index] == "Gryffindor"))))
		b4 = b4 - a * (random_student * (sigmoide(random_student @ b4) - (int(index[random_index] == "Hufflepuff"))))
		# print(b1)
		if i == 10000 :
			break
		i += 1
	weight = pd.DataFrame({"Ravenclaw": b1, "Slytherin": b2, "Gryffindor":b3, "Hufflepuff":b4, "mean":mean, "std":std})
	# weight = pd.DataFrame({"Ravenclaw": b1, "Slytherin": b2, "Gryffindor":b3, "Hufflepuff":b4})
	weight.to_csv("weight_stoch.csv")
	return

if __name__ == "__main__":
	try:
		main(sys.argv[1])
	except Exception:
		traceback.print_exc()
