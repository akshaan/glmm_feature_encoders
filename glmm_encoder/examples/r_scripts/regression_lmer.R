if(!require(lme4)){
    install.packages("lme4")
    library(lme4)
}
library(lme4)

data = read.csv('../data/toy_regression_data.csv', header = TRUE)
fit <- glmer(y ~ 1 + (1 | x), data=data)
fit
