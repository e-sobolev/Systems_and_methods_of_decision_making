library(MASS)
trainingSampleNormalization <- function(xl)
{
  n <- dim(xl)[2] - 1 
  for(i in 1:n)
  {
    xl[, i] <- (xl[, i] - mean(xl[, i])) / sd(xl[, i])
  }
  return (xl)
}

## ���������� ������� ��� �� -1 ��� w0
trainingSamplePrepare <- function(xl)
{
  l <- dim(xl)[1]
  n <- dim(xl)[2] - 1
  xl <- cbind(xl[, 1:n], seq(from = -1, to = -1,
                             length.out = l), xl[, n + 1])
}

## ������������ ������� ������
lossQuad <- function(x)
{
  return ((x-1)^2)
}

## �������������� �������� ��� ADALINE
sg.ADALINE <- function(xl, eta = 1, lambda = 1/6)
{
  l <- dim(xl)[1]
  n <- dim(xl)[2] - 1
  w <- c(1/2, 1/2, 1/2)
  iterCount <- 0
  
  ## initialize Q
  Q <- 0
  for (i in 1:l)
  {
    ## calculate the scalar product <w,x>
    wx <- sum(w * xl[i, 1:n])
    ## calculate a margin
    margin <- wx * xl[i, n + 1]
    
    Q <- Q + lossQuad(margin)
  }
  
  repeat
  {
    ## calculate the margins for all objects of the training sample
    margins <- array(dim = l)
    for (i in 1:l)
    {
      xi <- xl[i, 1:n]
      yi <- xl[i, n + 1]
      
      margins[i] <- crossprod(w, xi) * yi
    }
    
    ## select the error objects
    errorIndexes <- which(margins <= 0)
    
    if (length(errorIndexes) > 0)
    {
      # select the random index from the errors
      i <- sample(errorIndexes, 1)
      iterCount <- iterCount + 1
      
      xi <- xl[i, 1:n]
      yi <- xl[i, n + 1]
      
      ## calculate the scalar product <w,xi>
      wx <- sum(w * xi)
      
      ## make a gradient step
      margin <- wx * yi
      
      ## calculate an error
      ex <- lossQuad(margin)
      eta <- 1 / sqrt(sum(xi * xi))
      w <- w - eta * (wx - yi) * xi
      
      ## Calculate a new Q
      Qprev <- Q
      Q <- (1 - lambda) * Q + lambda * ex
    }
    else
    {
      break
    }
  }
  return (w)
}

# ���-�� �������� � ������ ������
ObjectsCountOfEachClass <- 500
## ���������� ��������� ������

Sigma1 <- matrix(c(20, 0, 0, 10), 2, 2)
Sigma2 <- matrix(c(10, 0, 0, 5), 2, 2)

xy1 <- mvrnorm(n=ObjectsCountOfEachClass, c(-5, 0), Sigma1)
xy2 <- mvrnorm(n=ObjectsCountOfEachClass, c(15, -10),
               Sigma2)
xl <- rbind(cbind(xy1, -1), cbind(xy2, +1))
colors <- c(rgb(255/255, 255/255, 0/255), "white",
            rgb(0/255, 200/255, 0/255))
## ������������ ������
xlNorm <- trainingSampleNormalization(xl)
xlNorm <- trainingSamplePrepare(xlNorm)
## ����������� ������
## ADALINE
plot(xlNorm[, 1], xlNorm[, 2], pch = 21, bg = colors[xl[,3]
                                                     + 2], asp = 1, xlab = "x1", ylab = "x2", main = "ADALINE")
w <- sg.ADALINE(xlNorm)
abline(a = w[3] / w[2], b = -w[1] / w[2], lwd = 3, col =
         "red")