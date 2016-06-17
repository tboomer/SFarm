# Script to read in and process images into a matrix that is fed to MXNet to
# train a neural network.

library(imager)
library(readr)
library(caret)
library(dplyr)

# Set resizing parameters
img_row = 48; img_col = 64
# Read driver data
driver_df <- read_csv('./input/driver_imgs_list.csv')

# Separate training and validation data sets
set.seed(510)
index <- createDataPartition(driver_df$classname, .85, list = FALSE, times = 1)
train_driver <- driver_df[index,]
val_driver <- driver_df[-index,]

# Take the driver and class data as input. Read each image into a matrix and resize. 
# Output result as data.frame with the label as first column. 

# Function that takes the readr image file and return edges calculated from the
# gradient.
find_edge <- function(image) {
     dx <- imgradient(image, 'x')
     dy <- imgradient(image, 'y')
     sqrt(dx^2 + dy^2)
}

summarize_img_stats <- function(path) {
     
     img_df <- load.image(path) %>% as.data.frame()
     img_df <- mutate(mutate(img_df, channel=factor(cc,labels=c('R','G','B'))))
     unlist(tapply(img_df$value, img_df$channel, summary))
}

get_BW_image_pixels <- function(path) {
     img <- load.image(path) %>%
          grayscale() %>%
          resize(size_x = img_row, size_y = img_col, interpolation_type = 3) %>%
          # find_edge() %>%
          as.vector()
}

get_color_image_pixels <- function(path) {
     img <- load.image(path) %>%
          resize(size_x = img_row, size_y = img_col, interpolation_type = 3) %>%
          # find_edge() %>%
          as.vector()
}

process_train_images <- function(data){
     X_matrix <- matrix(nrow = nrow(data), ncol = img_row * img_col) # *3 if color
     # X_matrix <- matrix(nrow = nrow(data), ncol = 18)

     # Loop over driver data to load image data
     for(i in 1:nrow(data)) {
               path <- paste('./input/train/', driver_df$classname[i], '/',
                             driver_df$img[i], sep = "")
               
          X_matrix[i, ] <- get_BW_image_pixels(path)
          
          if(i %% 1000 == 0) cat("Iteration ", i, '\n')
     }
     y_data <- as.factor(data$classname)
     driver <- as.factor(data$subject)
     data.frame(label = y_data, driver = driver, X_matrix)
}

process_test_images <- function(){
          test_files <- list.files('./input/test')
     X_matrix <- matrix(nrow = length(test_files), ncol = img_row * img_col)
     
     for(i in 1:length(test_files)) {
          path <- paste('./input/test/', test_files[i], sep = "")
          
          img <- load.image(path) %>%
               # grayscale() %>%
               resize(size_x = img_row, size_y = img_col) %>%
               as.vector()
          
          X_matrix[i, ] <- img
          
          if(i %% 1000 == 0) cat("Iteration ", i, '\n')
     }
     return(X_matrix)
}




# Process Data
Sys.time()
print("processing train images")
train <- process_train_images(train_driver)
print("processing val images")
val <- process_train_images(val_driver)
# print("processing test images")
# test <- process_test_images()
Sys.time()

# colname <- c('label', 'driver', 'R1Q', 'RMean', 'RMed', 'R3Q',
#              'G1Q', 'GMean', 'GMed', 'G3Q', 'B1Q', 'BMean', 'BMed', 'B3Q')
# train <- select(train, c(1:2, 4:7, 10:13, 16:19))
# names(train) <- colname
# 
# val <- select(val, c(1:2, 4:7, 10:13, 16:19))
# names(val) <- colname

# Save data
save(train, val, file = "./cache/gray3072B.RData")

