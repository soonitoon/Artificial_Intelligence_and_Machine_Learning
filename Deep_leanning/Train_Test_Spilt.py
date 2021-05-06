import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder

iris = sns.load_dataset("iris")

X = iris.iloc[:,0:4].values
y = iris.iloc[:,4].values

encoder = LabelEncoder()
y1 = encoder.fit_transform(y)
Y = pd.get_dummies(y1).values