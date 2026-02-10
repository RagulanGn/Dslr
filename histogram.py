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

	for col in df3.columns:
		plt.subplot()
		plt.hist(df_Ravenclaw[col])
	plt.show()

	# print(df3["Ravenclaw"])
	return

if __name__ == "__main__":
    # check arg
    main(sys.argv[1])