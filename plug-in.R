# Center of normal distribution
estimate_mu <- function(objects) {
  rows <- dim(objects)[1]
  cols <- dim(objects)[2]
  mu <- matrix(NA, 1, cols)
  for (col in 1:cols) {
    mu[1, col] = mean(objects[ ,col])
  }
  return(mu)
}

# Covariation matrix of normal distribution
estimate_cov_matrix <- function(objects, mu) {
  rows <- dim(objects)[1]
  cols <- dim(objects)[2]
  sigma <- matrix(0, cols, cols)
  for (i in 1:rows) {
    sigma <- sigma + (t(objects[i, ] - mu) %*% (objects[i, ] - mu)) / (rows - 1)
  }
  return(sigma)
}

# Get coefficients of plug-in
get_coeffs <- function(mu1, sigma1, mu2, sigma2) {
  # Line equation: a*x1^2 + b*x1*x2 + c*x2 + d*x1 + e*x2 + f = 0
  # Inverse matrices
  invSigma1 <- solve(sigma1) 
  invSigma2 <- solve(sigma2)
  f <- log(abs(det(sigma1))) - log(abs(det(sigma2))) + mu1 %*% invSigma1 %*% t(mu1) - mu2 %*% invSigma2 %*% t(mu2);
  alpha <- invSigma1 - invSigma2
  a <- alpha[1, 1]
  b <- 2 * alpha[1, 2]
  c <- alpha[2, 2]
  beta <- invSigma1 %*% t(mu1) - invSigma2 %*% t(mu2)
  d <- -2 * beta[1, 1]
  e <- -2 * beta[2, 1]
  return(c("x^2" = a, "xy" = b, "y^2" = c, "x" = d, "y" = e, "1" = f))
}

# Count of objects in each class
objects_count <- 10 

# Generation of test data
Sigma1 <- matrix(c(5, 0, 0, 1), 2, 2)
Sigma2 <- matrix(c(5, 0, 0, 1), 2, 2)
Mu1 <- c(-1, 0)
Mu2 <- c(3, 0)
xy1 <- mvrnorm(n = objects_count, Mu1, Sigma1)
xy2 <- mvrnorm(n = objects_count, Mu2, Sigma2)

# Assembling 2 classes in one sample xl
xl <- rbind(cbind(xy1, 1), cbind(xy2, 2))

# Drawing the training sample
colors <- c("orange", "green3")
plot(xl[ , 1], xl[ , 2], pch = 21, bg = colors[xl[ ,3]], asp = 1, xlab = "x", ylab = "y")

# Evaluation
objects_first <- xl[xl[,3] == 1, 1:2]
objects_second <- xl[xl[,3] == 2, 1:2]
mu1 <- estimate_mu(objects_first)
mu2 <- estimate_mu(objects_second)
sigma1 <- estimate_cov_matrix(objects_first, mu1)
sigma2 <- estimate_cov_matrix(objects_second, mu2)
coeffs <- get_coeffs(mu1, sigma1, mu2, sigma2)

# Drawing of discriminant function
x <- y <- seq(-10, 20, len = 100)
z <- outer(x, y, function(x, y) coeffs["x^2"]*x^2 + coeffs["xy"]*x*y + coeffs["y^2"]*y^2 + coeffs["x"]*x + coeffs["y"]*y + coeffs["1"])
contour(x, y, z, levels = 0, drawlabels = FALSE, lwd = 2.5, col = "red", add = TRUE)