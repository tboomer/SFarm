# This script trains a neural net 

library(caret)
source('./Rscripts/log_loss.R')

train_data <- readRDS('./cache/train_data_0510.RDS')
X_data <- train_data[[1]]
y_data <- train_data[[2]]

X_data <- t(X_data)
y_data <- as.factor(y_data)
levels(y_data) = c('c0', 'c1', 'c2', 'c3', 'c4', 'c5',
           'c6', 'c7', 'c8', 'c9')

val_data <- readRDS('./cache/val_data_0510.RDS')
V_data <- val_data[[1]]
w_data <- val_data[[2]]

V_data <- t(V_data)
w_data <- as.factor(w_data)
levels(w_data) = c('c0', 'c1', 'c2', 'c3', 'c4', 'c5',
                   'c6', 'c7', 'c8', 'c9')

Sys.time()
mlp_mod <- train(x = X_small, y = y_small,
                 method = 'mlp')
Sys.time()

pred <- predict(mlp_mod, newdata = V_data, type = 'prob')
