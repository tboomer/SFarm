# XGBoost model

library(xgboost)
source('./Rscripts/log_loss.R')


load('./cache/sum_stats.RData')


# Train XGBoost Model
dtrain<-xgb.DMatrix(data = data.matrix(train[,3:14]),           # train[,-1]
                    label = as.numeric(train$driver)-1)          # train$label
watchlist<-list(train = dtrain)
set.seed(0114)
xgb_param <- list(objective         = 'multi:softprob',
                  booster           = 'gbtree',
                  num_class         = 26,
                  eta               = 0.1,
                  max_depth         = 6,
                  min_child_weight  = 1,
                  subsample         = 0.8,
                  colsample_by_tree = 0.8,
                  lambda            = 0,
                  alpha             = 0)

xgb_mod <- xgb.train(params         = xgb_param,
                     data           = dtrain,
                     nround         = 100,
                     watchlist      = watchlist,
                     eval_metric    = 'merror',
                     maximize       = FALSE,
                     early.stop.round = 100)

val_pred <- predict(xgb_mod, data.matrix(val[, 3:14]))           # val[, -1])
val_pred <- matrix(val_pred, nrow = length(val_pred)/26, 
                      ncol = 26, byrow = TRUE)

val_pred_class <- apply(val_pred, 1, which.max)
table(val$label, val_pred_class)

