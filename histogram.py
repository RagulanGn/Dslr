import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a histogram based on a tab of std by Hogwart house
# Step 1 Reorder the tab based on each Hogwart House
# 		if df[Hogwart House] == "house_name":
# 			put inside new tab ?
# Step 2 Calculate the the std of each one
# Step 3 Display the histogram ( std by Hogwart House)


def main(path:str):
	df = pd.read_csv(path, index_col=0)
	df2 = df.set_index("Hogwarts House", append=True) #Add Hogwart House as index
	df3 = df2.select_dtypes(include=np.number).dropna(axis=1, how='all') #Drop every NaN columns
	df_Ravenclaw = df3.xs("Ravenclaw", level=1, drop_level=False)
	df_Slytherin = df3.xs("Slytherin", level=1, drop_level=False)
	df_Hufflepuff = df3.xs("Hufflepuff", level=1, drop_level=False)
	df_Gryffindor = df3.xs("Gryffindor", level=1, drop_level=False)

	i = 0
	fig, axs = plt.subplots(4,4)
	for ax in axs.flat[13:]:
		ax.set_visible(False)
 
	for col in df3.columns:
		axs[i // 4, i % 4].hist(df_Ravenclaw[col], alpha=0.7)
		axs[i // 4, i % 4].hist(df_Slytherin[col], alpha=0.7)
		axs[i // 4, i % 4].hist(df_Hufflepuff[col], alpha=0.7)ds 
		axs[i // 4, i % 4].hist(df_Gryffindor[col], alpha=0.7)
		axs[i // 4, i % 4].set_title(col)
		i += 1
  
	for ax in axs[-2, 1:]:
		ax.set_xlabel("Grade")
	axs[-1,0].set_xlabel("Grade")

	for ax in axs[:, 0]:       # left column → y labels
		ax.set_ylabel("Number of students")
	plt.tight_layout()
	# plt.show()
	plt.savefig("figure.png")
	return

if __name__ == "__main__":
	if (len(sys.argv) != 2):
		sys.exit("Wrong number of arguments")
	try:
		main(sys.argv[1])
	except:
		print("File not found")
    # Better check arg ??
