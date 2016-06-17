# Miscellaneous code snippits.

# Look at sample photos by driver
view_images <- function(data, rows){
     
     # Loop over driver data and plot images
     for(i in 1:nrow(data)) {
          path <- paste('./input/train/', driver_df$classname[samp[i]], '/',
                        driver_df$img[samp[i]], sep = "")
          
          img <- load.image(path) # %>%
          plot(img)
               # grayscale() %>%
               # resize(size_x = img_row, size_y = img_col, interpolation_type = 3) %>%
               # find_edge() %>%
               # as.vector()
          
          # X_matrix[i, ] <- img
          # if(i %% 1000 == 0) cat("Iteration ", i, '\n')
     }
     # y_data <- as.factor(data$classname)
     # driver <- as.factor(data$subject)
     # data.frame(label = y_data, driver = driver, X_matrix)
}

driver <- 'p012'
i <- sample(1 : sum(driver_df$subject == driver), 20)
samp <- which(driver_df$subject == driver)[i]
view_images(driver_df[samp,], samp)
#-------------------------------------------------------------------------------

# Plot mean values
i <- 555
path <- paste('./input/train/', driver_df$classname[i], '/',
driver_df$img[i], sep = "")

img <- load.image(path)
plot(img)
img_df <- as.data.frame(img)
img_mut <- mutate(mutate( img_df,channel=factor(cc,labels=c('R','G','B'))))
ggplot(img_mut,aes(value,col=channel))+geom_histogram(bins=30)+facet_wrap(~ channel)
tapply(img_mut$value, img_mut$channel, mean)
#-------------------------------------------------------------------------------

colorMeans <- group_by(tr[,-1], driver) %>% summarize_each(funs(mean))
ggplot(colorMeans, aes(x=RMean, y=GMean, color = driver)) + geom_point()
#-------------------------------------------------------------------------------

library(rpart)

tree <- rpart(driver ~ ., method = 'class', data = train)
val_pred <- predict(tree, newdata=val, type = 'class')

