# Apriori

# Data Preprocessing
# install.packages('arules')
library(arules)
dataset = read.csv('~/Dropbox/github/machine_learning_udemy/machine-learning-udemy/part-5-association-rule-learning/Apriori/Market_Basket_Optimisation.csv', header = FALSE)
dataset = read.transactions('~/Dropbox/github/machine_learning_udemy/machine-learning-udemy/part-5-association-rule-learning/Apriori/Market_Basket_Optimisation.csv', sep = ',', rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)

# Training Apriori on the dataset
rules = apriori(data = dataset, parameter = list(support = 0.004, confidence = 0.2))

# Visualising the results
inspect(sort(rules, by = 'lift')[1:10])
