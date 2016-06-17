# Load pre-processed data and run mxnet model.

library(mxnet)
source('./Rscripts/log_loss.R')

train <- readRDS('./cache/train_data_0510.RDS')
val <- readRDS('./cache/val_data_0510.RDS')

X_train <- train[[1]]
y_train <- train[[2]]

X_val <- val[[1]]
y_val <- as.factor(val[[2]])

# Take subset of training data
set.seed(595)
sample_index <- sample(1:ncol(X_train), 1000)
X_small <- X_train[, sample_index]
y_small <- y_train[sample_index]


# Configure Network
data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden = 832)
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=64)
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=10)
softmax <- mx.symbol.SoftmaxOutput(fc3, name="sm")

# Train network
start <- Sys.time()
start

devices <- mx.cpu()

mx.set.seed(12)
model <- mx.model.FeedForward.create(softmax, X=X_small, y=y_small,
                                     ctx=devices, num.round=10, array.batch.size=100,
                                     learning.rate=0.1, momentum=0.9,  eval.metric=mx.metric.accuracy,
                                     initializer=mx.init.uniform(0.07),
                                     epoch.end.callback=mx.callback.log.train.metric(100))

end <- Sys.time()
end - start