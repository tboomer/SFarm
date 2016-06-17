# This function converts pixel values stored in data frame columns created by the
# process_images.R script back into cimg format for plotting or further manipulation
# with readr commands. img is the subset of data frame columns that contain pixel values
# x is the number of imabe rows and y is the number of image columns.
# img should be a single row of the data frame.

require(imager)

to_image <- function(img, x, y, z = 1) {
     if(x * y *z != length(img)) stop('Dimensions for x and y do not match vector length')
     img_vector <- as.vector(img, 'numeric')
     img_array <- array(img_vector, c(x, y, z = z))
     as.cimg(img_array)
     
}
