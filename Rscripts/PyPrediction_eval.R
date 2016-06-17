

library(readr)
source('./Rscripts/log_loss.R')


pred <- read_csv('./cache/predictions.csv')
yval <- read_csv('./cache/yvals.csv')

colnames(yval) <- c('1','2','3','4','5','6','7','8','9', '10')
colnames(pred) <- c('1','2','3','4','5','6','7','8','9', '10')

y <- apply(yval, 1, function(x) which(x == 1))

ypred <- apply(pred, 1, function(x) which(x == max(x)))
table(y, ypred)
logloss(y, pred)
