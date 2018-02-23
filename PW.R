colors <- c("setosa" = "red", "versicolor" = "green3", "virginica" = "blue")
plot(NULL, NULL, type = "l", xlim = c(min(iris[, 3]), max(iris[, 3])), ylim = c(min(iris[, 4]), max(iris[, 4])), xlab = 'Petal.Length', ylab = 'Petal.Width')
title("Parzen Window")
euclideanDistance <- function(u, v) {
  sqrt(sum((u - v)^2))
}

CoreGaussian <- function(r) {
  return(((2*pi)^(-0.5))*exp(-0.5*(r^2)))
}

h <- 0.02

xl <- iris[, 3:5]
l <- dim(xl)[1] 
n <- dim(xl)[2] - 1 

X <- seq(from = min(iris[, 3]), to = max(iris[, 3]), by = 0.1)
Y <- seq(from = min(iris[, 4]), to = max(iris[, 4]), by = 0.1)

for(i in X) {
  for(j in Y) {
    point <- c(i, j)
    distances_weighted <- matrix(NA, l, 2)
    for (p in 1:l) {
      distances_weighted[p, 1] <- euclideanDistance(xl[p, 1:n], point)
      r <- distances_weighted[p, 1] / h
      distances_weighted[p, 2] <- CoreGaussian(r)
    }
    classes <- data.frame(distances_weighted[ , 1], distances_weighted[ , 2], xl[ , 3]) 
    colnames(classes) <- c("Distances", "Weights", "Species")
    
    sumSetosa <- sum(classes[classes$Species == "setosa", 2])
    sumVersicolor <- sum(classes[classes$Species == "versicolor", 2])
    sumVirginica <- sum(classes[classes$Species == "virginica", 2])
    answer <- matrix(c(sumSetosa, sumVersicolor, sumVirginica), 
                     nrow = 1, ncol = 3, byrow = T, list(c(1), c('setosa', 'versicolor', 'virginica')))
    points(point[1], point[2],  pch = 21, bg = "white", col = colors[which.max(answer)])
  }
}

for (i in 1:l) {
  points(iris[i, 3], iris[i, 4],  pch = 21, bg = colors[iris$Species[i]], col = colors[iris$Species[i]])
}

legend("bottomright", c("virginica", "versicolor", "setosa"), pch = c(15,15,15), col = c("blue", "green3", "red"))