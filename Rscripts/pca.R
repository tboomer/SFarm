# Script to apply PCA to training data set

source('./Rscripts/log_loss.R')


load('./cache/gray4800C.RData')

# Run PCA
pca <- prcomp(train[, -1], # Training data less category variable
                 center = TRUE,
                 scale = TRUE)

screeplot(pca, type = "lines", col = 2)
# summary(pca)

# After reviewing summary, identify how many variables account for 95% of variance.
trainPCA <- pca$x[ , 1:100]
train <- data.frame(label = train$label, trainPCA)

# Transform validation set into principal components
valPCA <- predict(pca, newdata = val[, -1])
valPCA <- valPCA[ , 1:100]
val <- data.frame(label = val$label, valPCA)

save(train, val, file = './cache/PCA100_4800C.RData')

#-------------------------------------------------------------------------------
# Transform test set into principal components
# testfactorvar <- which(unlist(lapply(testdata, class)) == "factor")
# testdata <- select(testdata, -testfactorvar)
testPCA <- predict(pca, newdata = testpredictors)
testPCA <- testPCA[ , 1:5]

trainLabel <- train$QuoteConversion_Flag
valLabel <- val$QuoteConversion_Flag

save(trainPCA, valPCA, testPCA,
     file = "./DataFiles/PCAPredictors3.RData")
