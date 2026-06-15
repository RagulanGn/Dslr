import sys
import pandas as pd
import numpy as np
from describe import ft_mean, ft_std

# Descente de gradient => Fonction a minimiser => Sigmoide : 1 /(1 - exp(-lambda * x)) Avec x = a * P1 + b * P2 + c * P3 + ... , PX les parametres
# Regression logistique 1 vs all => 4 sigmoide independante

# Courbe sigmoide => Double asymptote horinzontale a y=1 et y=-1

def sigmoide(x):
	""" Return the value of the sigmoide function at point x """
	x_safe = np.clip(x, -500, 500)
	return (1/(1 + np.exp(-x_safe)))

def gradient_descent(df, lr, epochs):
	b1 = b2 = b3 = b4 = np.zeros(len(df.columns))																			#Initialize all b vector of size = len(features) with 0
	i = 0
	n = len(df.index)																										#Number of line in our dataset => Allow us to calculate the mean of the error in each step
	index = df.index.get_level_values("Hogwarts House").to_numpy()															#Numpy array of indexes (Hogwart House of each student)
	while (i < epochs) :
		b1 = b1 - (lr/n) * (df.T @ (sigmoide(df @ b1) - (index == "Ravenclaw").astype(int)))									#Correction of parameters corresponding to Ravenclaw
		b2 = b2 - (lr/n) * (df.T @ (sigmoide(df @ b2) - (index == "Slytherin").astype(int)))
		b3 = b3 - (lr/n) * (df.T @ (sigmoide(df @ b3) - (index == "Gryffindor").astype(int)))
		b4 = b4 - (lr/n) * (df.T @ (sigmoide(df @ b4) - (index == "Hufflepuff").astype(int)))
		i += 1
	return [b1, b2, b3 , b4]

def SGD(df, lr, epochs):
	b1 = b2 = b3 = b4 = np.zeros(len(df.columns))																			#Initialize all b vector of size = len(features) with 0
	i = 0
	index_house = df.index.get_level_values("Hogwarts House").to_numpy()															#Numpy array of indexes (Hogwart House of each student)
	index = np.arange(len(df.index))
	while (i < epochs) :
		np.random.shuffle(index)
		for j in index:
			random_student = df.iloc[j]
			b1 = b1 - lr * (random_student * (sigmoide(random_student @ b1) - (int(index_house[j] == "Ravenclaw"))))
			b2 = b2 - lr * (random_student * (sigmoide(random_student @ b2) - (int(index_house[j] == "Slytherin"))))
			b3 = b3 - lr * (random_student * (sigmoide(random_student @ b3) - (int(index_house[j] == "Gryffindor"))))
			b4 = b4 - lr * (random_student * (sigmoide(random_student @ b4) - (int(index_house[j] == "Hufflepuff"))))
		i += 1
	return [b1, b2, b3 , b4]

def minibatch(df, lr, epochs):
	b1 = b2 = b3 = b4 = np.zeros(len(df.columns))																			#Initialize all b vector of size = len(features) with 0
	i = 0
	batch_size = 64																											#Size of batch
	while (i < epochs) :
		df = df.sample(frac=1)
		for j in range(0, len(df.index), batch_size):
			batch = df[j: j + batch_size]
			n = len(batch)
			index_batch = batch.index.get_level_values("Hogwarts House").to_numpy()
			b1 = b1 - (lr/n) * (batch.T @ (sigmoide(batch @ b1) - (index_batch == "Ravenclaw").astype(int)))		#Correction of parameters corresponding to Ravenclaw
			b2 = b2 - (lr/n) * (batch.T @ (sigmoide(batch @ b2) - (index_batch == "Slytherin").astype(int)))		#Correction of parameters corresponding to Ravenclaw
			b3 = b3 - (lr/n) * (batch.T @ (sigmoide(batch @ b3) - (index_batch == "Gryffindor").astype(int)))		#Correction of parameters corresponding to Ravenclaw
			b4 = b4 - (lr/n) * (batch.T @ (sigmoide(batch @ b4) - (index_batch == "Hufflepuff").astype(int)))		#Correction of parameters corresponding to Ravenclaw
		i += 1
	return [b1, b2, b3 , b4]

def main(lr=1e-3, epochs=100):
	try:
		if len(sys.argv) != 3:
			print("No optimizer or dataset path , usage : python program_name optimizer dataset_path")
			print("Optimizer available : SGD, minibatch, gradientdescent (default)")
			sys.exit(1)
		path = sys.argv[2]
		df = pd.read_csv(path, index_col=0)
		df = df.set_index("Hogwarts House", append=True)

	except (FileNotFoundError, PermissionError, pd.errors.EmptyDataError, pd.errors.ParserError):
		print("Incorrect path or file.")
		sys.exit(1)

	df = df.select_dtypes(include=np.number).dropna(axis=1, how='all')														#Drop every NaN columns 
	df = df.drop(columns=["Arithmancy", "Care of Magical Creatures"], errors='ignore')										#Drop 2 columns because they are not vector of any information but create noise
	df = df.fillna(df.mean())																								#Replace NaN with mean
 
	mean = ft_mean(df)
	std = ft_std(df)
	df = (df - mean) / std																									#Standardization to avoid overflow
	df.insert(0, 'bias', 1)																									#Bias to not force our model to pass by origin

	if (sys.argv[1] == 'SGD'):
		b1, b2, b3, b4 = SGD(df, lr, epochs)
	elif (sys.argv[1] == 'minibatch'):
		b1, b2, b3, b4 = minibatch(df, lr, epochs)
	else:
		b1, b2, b3, b4 = gradient_descent(df, lr, epochs)
	weight = pd.DataFrame({"Ravenclaw": b1, "Slytherin": b2, "Gryffindor": b3, "Hufflepuff": b4, "mean": mean, "std": std}) # Create a dataframe with the 4 weight combined + mean and standard deviation
	weight.to_csv("weight.csv") #Save to csv
	return

if __name__ == "__main__":
	try:
		main(lr=1e-3, epochs=100)
	except Exception as e:
		sys.exit(f"Error : {e}")