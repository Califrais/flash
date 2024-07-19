# Title     : TODO
# Objective : TODO
# Created by: user
# Created on: 09/12/2021

defaultW <- getOption("warn")
options(warn = -1)
library('joineRML')

load <- function() {
    set.seed(0)
    beta <- rbind(c(0.05, .01, .01, .01),
    c(-0.2, -.1, -0.05, -0.1))
    D <- diag(4)
    
    D[1, 1] <- D[2, 2] <- D[3, 3] <- 0.5
    D[1, 2] <- D[2, 1] <- D[3, 4] <- D[4, 3] <- 0.1
    D[1, 3] <- D[3, 1] <- 0.01
    sim <- simData(n = 250, beta = beta, D = D, sigma2 = c(0.05, 0.05),
                   censoring = TRUE, censlam = exp(-3), gamma.x =  c(-2., -1.), gamma.y = c(-.02, .01), ntms = 30, 
                   theta0 = .1, theta1 = .1, truncation=TRUE)
    data <- merge(sim$survdat,sim$longdat,by="id")
    
    data$T_survival <- data$survtime
    data$delta <- data$cens
    data$T_long <- data$ltime

    return(data)
}