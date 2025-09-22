import pandas as pd
import numpy as np
import seaborn as sns

a=np.array([10,12,13,45,17])
print(a)
print(f"Mean for the n-dimensional array: {a.mean()}")

data=sns.load_dataset('tips')
print(data.head(10))