# Run an SVM model

library(e1071)
library(doParallel)

source('./Rscripts/log_loss.R')

load('./cache/gray832C.RData')


cl <- makeCluster(detectCores())

set.seed(1214)
start <- Sys.time()
start
registerDoParallel(cl)
svmMod <- svm(x = train[, -1], y = train$label,
              scale = TRUE,
              type = "C-classification",
              kernel = "radial",
              #              degree = 3,
              #              nu = 0.1,
              cost = 1,
              probability = TRUE)

stopCluster(cl)
end <- Sys.time()
cat(end, end - start)

# Make prediction on val set
val_pred <- predict(svmMod, newdata = val[, -1], probability = TRUE)
table(val$label, val_pred)
val_prob <- attr(val_pred, "probabilities")
cat("Val log loss: ", logloss(val$label, val_prob))

# train_pred <- predict(svmMod, probability = TRUE)
# train_prob <- attr(train_pred, "probabilities")
# cat("Train log loss: ", logloss(train$label, train_prob))

