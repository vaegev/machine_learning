# polynomial regression

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$DependentVariable, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# fit linear regression
lin_reg = lm(formula = Salary ~ .,
             data = dataset)

# fit polynomial regression
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ .,
              data = dataset)
# vis lin reg
library(ggplot2)
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)),
            colour = 'blue') +
  ggtitle('lin reg') +
  xlab('level') +
  ylab('salary')

# vis poly reg

ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)),
            colour = 'blue') +
  ggtitle('poly reg') +
  xlab('level') +
  ylab('salary')

# predict lin reg
y_pred = predict(lin_reg, newdata = data.frame(Level=6.5))

# predict poly reg
y_pred = predict(poly_reg, newdata = data.frame(Level=6.5, 
                                                Level2 = 6.5^2, 
                                                Level3=6.5^3, 
                                                Level4=6.5^4))

