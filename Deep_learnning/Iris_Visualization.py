import seaborn as sns
import pandas as pd
import numpy as np

sns.set(style="ticks", color_codes=True)
iris = sns.load_dataset("iris") # sameple data
g = sns.pairplot(iris, hue="species", palette="husl")