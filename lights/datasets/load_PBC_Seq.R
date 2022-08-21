# Title     : TODO
# Objective : TODO
# Created by: user
# Created on: 09/12/2021

defaultW <- getOption("warn")
options(warn = -1)
library("JMbayes")

load <- function() {

    data(pbc2, package = "JMbayes")
    data <- pbc2
    data$T_long <- data$year
    data$T_survival <- data$years
    data$delta <- data$status2

    return(data)
}