# Calculate the log loss function to evaluate accuracy of competition submissions

# Function accepts a vector (x) of integer values and outputs a sparse matrix where
# nrow = length(vector), each column corresponds to a unique integer value in
# vector and there is a 1 in the corresponding row for each element of the vector.
integer_to_sparse <- function(x) {
     sparse <- matrix(0, length(x), max(x))
     values <- matrix(c(1:length(x), x), length(x), 2)
     sparse[values] <- 1
     return(sparse)
}

# Function computes the log loss of the probability calculated for the correct
# actual value including the adjustment of the minimum p value specified in the
# competition rules.
logloss <- function(actual, prediction){
     prediction <- prediction / apply(prediction, 1, sum) # Rescale probabilities to equal 1
     actual <- as.numeric(actual) # Convert label from factor to numeric {1:10}
     p <- rowSums(integer_to_sparse(actual) * prediction) # Make vector of p of correct values
     p <- sapply(p, function(x) max(min(x, (1-10^-15)), 10^-15)) # Apply minimum formula
     - sum(log(p)) / length(p) # sum log loss
}