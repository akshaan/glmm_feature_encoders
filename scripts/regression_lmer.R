data = read.csv('../data/toy_regression_data.csv', header = TRUE)

# install.packages('lme4')
library(lme4)
fit <- glmer(y ~ 1 + (1 | x), data=data)
fit
