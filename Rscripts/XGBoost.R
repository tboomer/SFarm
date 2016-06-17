# XGBoost model

library(xgboost)
source('./Rscripts/log_loss.R')


load('./cache/PCA1200C.RData')


# Train XGBoost Model
dtrain<-xgb.DMatrix(data = data.matrix(trainPCA),           # train[,-1]
                    label = as.numeric(y_train)-1)          # train$label
watchlist<-list(train = dtrain)
set.seed(0114)
xgb_param <- list(objective         = 'multi:softprob',
                  booster           = 'gbtree',
                  num_class         = 10,
                  eta               = 0.1,
                  max_depth         = 6,
                  min_child_weight  = 1,
                  subsample         = 0.8,
                  colsample_by_tree = 1,
                  lambda            = 0,
                  alpha             = 0)

xgb_mod <- xgb.train(params         = xgb_param,
                     data           = dtrain,
                     nround         = 200,
                     watchlist      = watchlist,
                     eval_metric    = 'mlogloss',
                     maximize       = FALSE,
                     early.stop.round = 100)

val_pred <- predict(xgb_mod, data.matrix(valPCA))           # val[, -1])
val_pred <- matrix(val_pred, nrow = length(val_pred)/10, 
                      ncol = 10, byrow = TRUE)
logloss(y_val, val_pred)                                        # val$label

val_pred_class <- apply(val_pred, 1, which.max)
table(y_val, val_pred_class)                                # val$label
