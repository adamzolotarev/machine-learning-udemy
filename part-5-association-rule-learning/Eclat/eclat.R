# Eclat

# Data Preprocessing
# install.packages('arules')
library(arules)
#dataset = read.csv('~/Dropbox/github/machine_learning_udemy/machine-learning-udemy/part-5-association-rule-learning/Eclat/Market_Basket_Optimisation.csv')
dataset = read.transactions('~/Dropbox/github/machine_learning_udemy/machine-learning-udemy/part-5-association-rule-learning/Eclat/Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Training Eclat on the dataset
rules = eclat(data = dataset, parameter = list(support = 0.004, minlen = 2))

# Visualising the results
inspect(sort(rules, by = 'support')[1:10])