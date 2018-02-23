colors <- c("setosa" = "red", "versicolor" = "green3", "virginica" = "blue")
plot(iris[, 3:4], pch = 21, bg = colors[iris$Species], col = colors[iris$Species], asp = 1)
title ("KwNN")

euclideanDistance <- function(u, v) {
  sqrt(sum((u - v)^2))
}

k <- 6
q <- 1 

xl <- iris[, 3:5]
l <- dim(xl)[1] 
n <- dim(xl)[2] - 1

X <- seq(min(iris[, 3]), max(iris[, 3]), 0.1)
Y <- seq(min(iris[, 4]), max(iris[, 4]), 0.1)

for(i in X) {
  for(j in Y) {
    z <- c(i, j)
    distances <- matrix(NA, l, 2)
    for(p in 1:l) {
      distances[p, ] <- c(p, euclideanDistance (xl[p, 1:n], z))
    }
    orderedxl <- xl[order(distances[ , 2]), ]
    weights <- c(NA)
    for(t in 1:l) {
      weights[t] <- q^t
    }
    orderedxl_weighted <- cbind(orderedxl, weights)
    classes <- orderedxl_weighted[1:k, (n + 1):(n + 2)] # number, name of class and weight
    sumSetosa <- sum(classes[classes$Species == "setosa", 2])
    sumVersicolor <- sum(classes[classes$Species == "versicolor", 2])
    sumVirginica <- sum(classes[classes$Species == "virginica", 2])
    answer <- matrix(c(sumSetosa, sumVersicolor, sumVirginica), 
                     nrow = 1, ncol = 3, byrow = T, list(c(1), c('setosa', 'versicolor', 'virginica')))
    points(z[1], z[2],  pch = 21, col = colors[which.max(answer)])
  }
}

for (i in 1:l) {
  points(iris[i, 3], iris[i, 4],  pch = 21, bg = colors[iris$Species[i]], col = colors[iris$Species[i]])
}

legend("bottomright", c("setosa", "versicolor", "virginica"), pch = c(15,15,15), col = c("red", "green3", "blue"))