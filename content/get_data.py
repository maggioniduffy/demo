import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

raw_data = pd.read_csv('europe.csv')
features = ['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']

x = raw_data.loc[:, features].values

data = StandardScaler().fit_transform(x)

# sum_of_rows = x.sum(axis=1)
# data = x / sum_of_rows[:, np.newaxis]
# print(data)
# data = x / np.linalg.norm(x)

# from sklearn.preprocessing import normalize
# data = normalize(x)
# print(data)
# print(data)