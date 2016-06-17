# Visualize distribution of predicted results

require(ggplot2)
require(readr)
require(reshape2)

filename <- './submissions/submission_0.160842539184_2016-06-09-02-46.csv'
submission <- read_csv(filename)
data <- melt(submission[, -11])

colnames(data) <- c('class', 'prob')

data <- filter(data, prob <.25)
ggplot(filter(data, prob <.25), aes(prob, fill = class)) + geom_histogram()
