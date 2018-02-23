colors <- c("setosa" = "red", "versicolor" = "green3", "virginica" = "blue")
plot(iris[, 3:4], pch = 21, bg = colors[iris$Species], col = colors[iris$Species], asp = 1)
title("6NN random points")

euclideanDistance <- function(u, v){
  sqrt(sum((u - v)^2))
}

sortObjectsByDist <- function(xl, z, metricFunction = euclideanDistance){
  l <- dim(xl)[1]
  n <- dim(xl)[2] - 1
  
  distances <- matrix(NA, l, 2)
  
  for (i in 1:l){
    distances[i, ] <- c(i, metricFunction(xl[i, 1:n], z))
  }
  
  orderedXl <- xl[order(distances[, 2]), ]
  
  return (orderedXl);
}

kNN <- function(xl, z, k){
  orderedXl <- sortObjectsByDist(xl, z)
  n <- dim(orderedXl)[2] - 1
  classes <- orderedXl[1:k, n + 1]
  counts <- table(classes)
  class <- names(which.max(counts))
  
  return (class)
}

painting <- function(x, y) {
  z <- c(x, y)
  xl <- iris[, 3:5]
  class <- kNN(xl, z, k=6)
  points(z[1], z[2], pch = 22, bg = colors[class], asp = 1)
}

testX <- runif(50, 1.0, 7.0)
testY <- runif(50, 0, 2.5)

for (i in 1:50){
  painting(testX[i], testY[i])
}

legend("bottomright", c("setosa", "versicolor", "virginica"), pch = c(15,15,15), col = c("red", "green3", "blue"))