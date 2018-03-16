# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
# First row is all products
dataset = pd.read_csv(
    '~/Dropbox/github/machine_learning_udemy/machine-learning-udemy/part-5-association-rule-learning/Apriori_Python/Market_Basket_Optimisation.csv', header=None)
transactions = []
# Needs to be imported in a specific way: list of lists
#
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

# Training Apriori on the dataset
from apyori import apriori
# support - say bought at least 3-4 times a day-
#           (7 times a week over total # of trans's because trans's are per week) 3*7/7500
# confidence - e.g., .8 would be too high - you would get only products purchased all the time
#
rules = apriori(transactions, min_support=0.003,
                min_confidence=0.2, min_lift=3, min_length=2)

# Visualising the results
results = list(rules)
# See items, support
# OrderedStatistic show confidence, lift
# item_base - if people buy item_base they are likely to buy item_add
# confidence - .29 means if they buy item_base they have 29% chance of buying item_add
# lift shows relevance of the rule
# if confidence is high, but lift is low, it probably means both products are just
# frequently bought in general, and not necessarily because they relate to each other.
print(results)
