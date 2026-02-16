import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create a histogram based on a tab of std by Hogwart house
# Step 1 Reorder the tab based on each Hogwart House
# 		if df[Hogwart House] == "house_name":
# 			put inside new tab ?
# Step 2 Calculate the the std of each one
# Step 3 Display the histogram ( std by Hogwart House)

def main(path:str):
	df = pd.read_csv(path, index_col=0)
	sns.pairplot(df, hue='Hogwarts House', diag_kind='kde')
	plt.savefig("pair_plot.png")


# def main(path:str):
# 	df = pd.read_csv(path, index_col=0)
# 	df2 = df.set_index("Hogwarts House", append=True) #Add Hogwart House as index
# 	df3 = df2.select_dtypes(include=np.number).dropna(axis=1, how='all') #Drop every NaN columns
# 	df_Ravenclaw = df3.xs("Ravenclaw", level=1, drop_level=False)
# 	df_Slytherin = df3.xs("Slytherin", level=1, drop_level=False)
# 	df_Hufflepuff = df3.xs("Hufflepuff", level=1, drop_level=False)
# 	df_Gryffindor = df3.xs("Gryffindor", level=1, drop_level=False)

# 	row_size = 13
# 	col_size = 13
# 	fig, axs = plt.subplots(row_size,col_size, figsize=(12,8))
# 	# for ax in axs.flat[13:]:
# 		# ax.set_visible(False)

# 	i = 0

# 	for colx in df3.columns:
# 		for coly in df3.columns:
# 			if colx == coly:
# 				pass
# 			else:
# 				axs[i // col_size, i % col_size].scatter(df_Ravenclaw[colx],df_Ravenclaw[coly], s=1)
# 				axs[i // col_size, i % col_size].scatter(df_Slytherin[colx],df_Slytherin[coly], s=1)
# 				axs[i // col_size, i % col_size].scatter(df_Hufflepuff[colx],df_Hufflepuff[coly], s=1)
# 				axs[i // col_size, i % col_size].scatter(df_Gryffindor[colx],df_Gryffindor[coly], s=1)
# 				axs[i // col_size, i % col_size].tick_params(labelbottom=False, labelleft=False)
# 			i += 1
# 	# plt.tight_layout()
# 	# plt.show()
# 	plt.savefig("pair_plot.png")
# 	return

if __name__ == "__main__":
	if (len(sys.argv) != 2):
		sys.exit("Wrong number of arguments")
	main(sys.argv[1])
	
	# try:
		# main(sys.argv[1])
	# except:
		# print("File not found")
    # Better check arg ??
