# Run deep learning model on data loaded from cache.

library(h2o)
library(readr)

source('./Rscripts/log_loss.R')

load('./cache/gray3072D.RData')


localH2O <- h2o.init(nthreads = -1, max_mem_size = "7g")  # Launch h2o instance

train.h2o <- as.h2o(train)
val.h2o <- as.h2o(val)

y.dep <- 1                  # 1
x.indep <- c(3:ncol(train)) # 2:ncol(train)

Sys.time()
dl_mod <- h2o.deeplearning(x = x.indep, y = y.dep,
                           training_frame = train.h2o,
                           epochs = 5,
                           nfolds = 5,
                           hidden = c(512, 512, 512),
                           l2 = 0.0001,
                           activation = 'Maxout')

pred <- as.data.frame(h2o.predict(dl_mod, newdata = val.h2o[, x.indep]))
dl_mod
table(val$label, pred[,1])            
logloss(val$label, pred[,2:11])
Sys.time()

# Close local h2o instance
h2o.shutdown(prompt = TRUE)

