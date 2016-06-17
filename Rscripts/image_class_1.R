# Script to read in and process images into a matrix that is fed to MXNet to
# train a neural network.

library(mxnet)
library(imager)
library(readr)
library(caret)

# Set resizing parameters
img_row = 26; img_col = 32
# Read driver data
driver_df <- read_csv('./input/driver_imgs_list.csv')

set.seed(510)
index <- createDataPartition(driver_df$classname, .85, list = FALSE, times = 1)
train <- driver_df[index,]
validate <- driver_df[-index,]

process_train_data <- function(data){
     X_matrix <- matrix(nrow = img_row * img_col, ncol = nrow(data))
     
     # Loop over driver data to load image data
     for(i in 1:nrow(data)) {
               path <- paste('./input/train/', driver_df$classname[i], '/',
                             driver_df$img[i], sep = "")
          
          img <- load.image(path) %>%
               grayscale() %>%
               resize(size_x = img_row, size_y = img_col) %>%
               as.vector()
          
          X_matrix[, i] <- img
          if(i %% 1000 == 0) cat("Iteration ", i, '\n')
     }
     y_data <- as.integer(as.factor(data$classname)) - 1
     list(X_matrix, y_data)
}

process_test_data <- function(){
          test_files <- list.files('./input/test')
     X_matrix <- matrix(nrow = img_row * img_col, ncol = length(test_files))
     
     for(i in 1:length(test_files)) {
          path <- paste('./input/test/', test_files[i], sep = "")
          
          img <- load.image(path) %>%
               grayscale() %>%
               resize(size_x = img_row, size_y = img_col) %>%
               as.vector()
          
          X_matrix[, i] <- img
          if(i %% 1000 == 0) cat("Iteration ", i, '\n')
     }
     return(X_matrix)
}


train_data <- process_train_data(train)
validate_data <- process_train_data(validate)
test_data <- process_test_data()

# Save train and val as data frames and test as matrix


